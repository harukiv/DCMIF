import torch
import gc
import tempfile
from PIL import Image
from model.JSRL.model_64 import *
from functorch.einops import rearrange
from transformers import BertModel, BertTokenizer
from torchvision.transforms.functional import to_pil_image

class SCGBlock(nn.Module):
    def __init__(self,
                 n_iter=1,
                 in_nc=1,
                 nc_x: List[int] = [64, 128, 256],
                 out_nc: int = 1,
                 nb: int = 1,
                 d_size: int = 5,
                 **kargs):
        super(SCGBlock, self).__init__()

        self.head1 = HeadNet(in_nc, nc_x, out_nc, d_size)
        self.head2 = HeadNet(in_nc, nc_x, out_nc, d_size)

        self.SC_body1 = SCBlock(in_nc = nc_x[0] + 1,
                               nc_x = nc_x,
                               nb = nb)
        self.SC_body2 = SCBlock(in_nc=nc_x[0] + 1,
                               nc_x=nc_x,
                               nb=nb)

        self.hypa_list: nn.ModuleList = nn.ModuleList()
        for _ in range(n_iter):
            self.hypa_list.append(HypaNet(in_nc=1, out_nc=4))

        self.n_iter = n_iter

    def forward(self, y1, y2, d, sigma):
        h, w = y1.size()[-2:]
        paddingBottom = int(ceil(h / 8) * 8 - h)
        paddingRight = int(ceil(w / 8) * 8 - w)
        y1 = F.pad(y1, [0, paddingRight, 0, paddingBottom], mode='circular')
        y2 = F.pad(y2, [0, paddingRight, 0, paddingBottom], mode='circular')
        N = y1.size(0)

        Y1 = torch.fft.fft2(y1)
        Y1 = torch.stack((Y1.real, Y1.imag), dim=-1)
        Y1 = Y1.unsqueeze(2)  # Y1:[N,3,1,H,W,2]
        Y2 = torch.fft.fft2(y2)
        Y2 = torch.stack((Y2.real, Y2.imag), dim=-1)
        Y2 = Y2.unsqueeze(2)  # Y1:[N,3,1,H,W,2]

        x1, _ = self.head1(y1, sigma)
        x2, _ = self.head2(y2, sigma)

        for i in range(self.n_iter):
            hypas = self.hypa_list[i](sigma, N)
            alpha_x = hypas[:, 0].unsqueeze(-1)
            beta_x = hypas[:, 1].unsqueeze(-1)

            x1 = self.SC_body1(x1, d, Y1, alpha_x, beta_x)
            x2 = self.SC_body2(x2, d, Y2, alpha_x, beta_x)

        return x1, x2

class TextPrompt(nn.Module):
    def __init__(self, text_dim=768, img_dim=64, max_token_len=256):
        super(TextPrompt, self).__init__()
        model_path = "../LLM/bert_base_uncased"
        self.bert = BertModel.from_pretrained(model_path, local_files_only=True)
        self.gamma_fc = nn.Linear(text_dim, img_dim)
        self.beta_fc = nn.Linear(text_dim, img_dim)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        self.max_token_len = max_token_len

    def generate_caption(self, vis, ir_p, LLM_model, LLM_tokenizer):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            if isinstance(vis, Image.Image):
                vis.save(tmp.name)
            else:
                raise TypeError("img must be a PIL.Image.Image object")

            image_path1 = tmp.name  # 临时图像路径

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            if isinstance(ir_p, Image.Image):
                ir_p = ir_p.convert("L")
                ir_p.save(tmp.name)
            else:
                raise TypeError("img must be a PIL.Image.Image object")

            image_path2 = tmp.name

        query = LLM_tokenizer.from_list_format([
            {'image': image_path1},
            {'text': 'This is the original visible-light image.'},
            {'image': image_path2},
            {'text': 'This is a synthetic infrared image generated from the visible image. Please analyze how realistic this infrared image appears, and suggest what modifications are needed to make it more physically accurate. Focus on thermal distribution, contrast, and texture fidelity in regions such as people, vehicles, buildings, trees, and roads.'}
        ])
        with torch.no_grad():
            response, _ = LLM_model.chat(tokenizer=LLM_tokenizer, query=query, history=None)

        del query
        gc.collect()
        torch.cuda.empty_cache()

        return response

    def forward(self, img_feature, vis, ir_p, LLM_model, LLM_tokenizer):
        vis = [to_pil_image(vis[i].clamp(0, 1)) for i in range(vis.shape[0])]
        ir_p = [to_pil_image(ir_p[i].clamp(0, 1)) for i in range(ir_p.shape[0])]
        prompt_list = []
        for i, (vis_img, ir_img) in enumerate(zip(vis, ir_p)):
            prompt_text = self.generate_caption(vis_img, ir_img, LLM_model, LLM_tokenizer)
            prompt_list.append(prompt_text)
        prompt_encoding = self.tokenizer(prompt_list,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=self.max_token_len,
                                         return_tensors='pt')
        device1 = torch.device("cuda:0")
        device2 = torch.device('cuda:1')
        input_ids = prompt_encoding['input_ids'].to(device2)
        attention_mask = prompt_encoding['attention_mask'].to(device2)

        self.bert.to(device2)
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.last_hidden_state[:, 0, :].detach()  # [B, text_dim]

        text_feat = text_feat.to(device1)
        # Step 2: 生成通道调制参数 γ 和 β
        gamma = self.gamma_fc(text_feat).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = self.beta_fc(text_feat).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]

        # Step 3: 执行通道调制
        modulated_feat = gamma * img_feature + beta  # [B, C, H, W]

        return modulated_feat

class PIGBlock(nn.Module):
    def __init__(self):
        super(PIGBlock, self).__init__()
        self.decoder = Decoder()
        self.attn1 = AttentionBase(dim=64)
        self.attn2 = ChannelAttention(dim=64)
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        sc = self.decoder(x)
        attn1 = self.attn1(sc)
        attn2 = self.attn2(sc)
        out = self.conv(attn1 + attn2)

        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = UNet()

    def forward(self, x):
        x_t = x.clone()
        x_cnn = self.decoder(x_t)
        output = x_cnn
        return output

class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale

def PromptMultiple(x, prompt):
    x_size = x.size()
    x = rearrange(x, 'b c h w -> b c (h w)')
    p = prompt.unsqueeze(0).repeat(x.shape[0], 1, 1)
    x = torch.matmul(p, x)
    x = rearrange(x, 'b c (h w) -> b c h w', h=x_size[2], w=x_size[3])
    return x

class MultiHead_Perception_Block(nn.Module):
    def __init__(self, in_nc, out_nc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        )
        self.attn_spatial = AttentionBase(dim=in_nc)
        self.attn_channel = ChannelAttention(dim=in_nc)
        self.cat = nn.Sequential(
            nn.Conv2d(3 * in_nc, in_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out_conv = self.conv(x)
        out_channel = self.attn_channel(x)
        out_spatial = self.attn_spatial(x)
        out = self.cat(torch.cat((out_conv, out_spatial, out_channel), dim=1))
        return out

class UNet(nn.Module):
    def __init__(self, in_nc = 64,
                 nc_x = [64, 128, 256],
                 out_nc = 64):
        super(UNet, self).__init__()
        self.encoder1 = MultiHead_Perception_Block(in_nc = in_nc, out_nc = in_nc)
        self.down1 = nn.Conv2d(in_nc, nc_x[1], kernel_size=2, stride=2, padding=0, bias=False)
        self.encoder2 = MultiHead_Perception_Block(in_nc = nc_x[1], out_nc = nc_x[1])
        self.down2 = nn.Conv2d(nc_x[1], nc_x[2], kernel_size=2, stride=2, padding=0, bias=False)

        self.body = MultiHead_Perception_Block(in_nc = nc_x[2], out_nc = nc_x[2])

        self.up2 = self.upsample_conv(nc_x[2], nc_x[1])
        self.decoder2 = MultiHead_Perception_Block(in_nc = nc_x[1], out_nc = nc_x[1])
        self.up1 = self.upsample_conv(nc_x[1], nc_x[0])
        self.decoder1 = MultiHead_Perception_Block(in_nc = nc_x[0], out_nc = nc_x[0])

        self.final = nn.Conv2d(nc_x[0], nc_x[0], kernel_size=1)

    def conv_block(self, in_nc, out_nc):
        return nn.Sequential(
            nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, padding=1)
        )

    def upsample_conv(self, in_nc, out_nc):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1)
        )

    def forward(self, x):
        enc1 = x
        enc2 = self.down1(self.encoder1(enc1) + enc1)
        enc3 = self.down2(self.encoder2(enc2) + enc2)
        body = self.body(enc3) + enc3
        dec3 = self.up2(body + enc3)
        dec3 = self.decoder2(dec3) + dec3
        dec2 = self.up1(dec3 + enc2)
        dec2 = self.decoder1(dec2) + dec2

        dec1 = dec2 + enc1
        output = self.final(dec1) + dec1

        return output

class SCFBlock(nn.Module):
    def __init__(self):
        super(SCFBlock, self).__init__()
        self.attn1 = WindowSelfAttention(dim=128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.attn2 = WindowSelfAttention(dim=64)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )

    def forward(self, vis_sc, ir_p_sc):
        f_sc = torch.cat((vis_sc, ir_p_sc), dim=1)
        f = self.conv1(self.attn1(f_sc)) + ir_p_sc
        f = self.conv2(self.attn2(f))
        return f


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, "H, W must be divisible by window_size"

        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)

        attn_out, _ = self.attn(x, x, x)
        attn_out = self.norm(attn_out + x)

        x = attn_out.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return x


def ycbcr_to_rgb_manual(ycbcr):
    y = ycbcr[:, 0:1, :, :]
    cb = ycbcr[:, 1:2, :, :] - 0.5
    cr = ycbcr[:, 2:3, :, :] - 0.5

    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = torch.cat([r, g, b], dim=1).clamp(0, 1)
    return rgb

def restore_rgb(y_batch, cb_batch, cr_batch):
    ycbcr_batch = torch.cat([y_batch, cb_batch, cr_batch], dim=1)

    rgb_batch = ycbcr_to_rgb_manual(ycbcr_batch)

    return rgb_batch


class PIVIF(nn.Module):
    def __init__(self,
                 **kargs):
        super(PIVIF, self).__init__()
        self.scg = SCGBlock()
        self.text_prompt = TextPrompt()
        self.pig = PIGBlock()
        self.scf = SCFBlock()
        self.tail = TailNet()

    def forward(self, ir, vis, d, sigma, cb, cr, LLM_model, LLM_tokenizer):
        device1 = torch.device('cuda:0')
        d = d.repeat(vis.size(0), 1, 1, 1, 1)
        sigma = sigma[:1]
        ir_sc , vis_sc = self.scg(ir, vis, d, sigma)
        sc = vis_sc.clone()
        ir_p_sc = self.pig(sc)
        ir_p = self.tail(ir_p_sc, d)
        sc = self.text_prompt(vis_sc, vis[:, :1, :, :], ir_p, LLM_model, LLM_tokenizer)
        sc = sc.to(device1)
        ir_p_sc = self.pig(sc)

        fuse_sc = self.scf(vis_sc, ir_p_sc)
        fuse = self.tail(fuse_sc, d)
        fuse_rgb = restore_rgb(fuse[:, :1, :, :], cb, cr)

        return fuse[:, :1, :, :], fuse_rgb

class Test(nn.Module):
    def __init__(self, **kargs):
        super(Test, self).__init__()
        self.scg = SCGBlock()
        self.text_prompt = TextPrompt()
        self.pig = PIGBlock()
        self.scf = SCFBlock()
        self.tail = TailNet()

    def forward(self, vis, d, sigma, cb, cr, LLM_model, LLM_tokenizer):
        device1 = torch.device('cuda:0')
        d = d.repeat(vis.size(0), 1, 1, 1, 1)
        sigma = sigma[:1]
        _, vis_sc = self.scg(vis, vis, d, sigma)
        sc = vis_sc.clone()
        ir_p_sc = self.pig(sc)
        ir_p = self.tail(ir_p_sc, d)
        sc = self.text_prompt(vis_sc, vis[:, :1, :, :], ir_p, LLM_model, LLM_tokenizer)
        sc = sc.to(device1)
        ir_p_sc = self.pig(sc)

        fuse_sc = self.scf(vis_sc, ir_p_sc)
        fuse = self.tail(fuse_sc, d)
        fuse_rgb = restore_rgb(fuse[:, :1, :, :], cb, cr)

        return fuse_rgb