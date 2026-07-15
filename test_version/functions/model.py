import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_features, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('P', self._generate_positional_encoding(max_len, num_features))

    def _generate_positional_encoding(self, max_len, num_features):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_features, 2) * (-np.log(10000.0) / num_features))
        positional_encoding = torch.zeros((1, max_len, num_features))
        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return positional_encoding



    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)   #  从位置编码张量 P 中提取与输入序列长度匹配的部分与 X 相加
        X = self.dropout(X)
        return X


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.PReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )
        self.skip_connection = nn.Sequential()
        if output_channel != input_channel or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        identity = self.skip_connection(x)
        out = identity + out
        out = self.relu(out)
        return out



class ResBlock_kan(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock_kan, self).__init__()
        self.conv = nn.Sequential(
            KANConv1DLayer(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            # nn.PReLU(),
            KANConv1DLayer(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )
        self.skip_connection = nn.Sequential()
        if output_channel != input_channel or stride != 1:
            self.skip_connection = nn.Sequential(
                KANConv1DLayer(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        identity = self.skip_connection(x)
        out = identity + out
        # out = self.relu(out)
        return out


class CNN_Transformer(nn.Module):
    '''
    input shape: (N, 3, 128)
    '''
    def __init__(self, config):
        super(CNN_Transformer, self).__init__()
        # CNN 部分
        self.layer1 = ResBlock(input_channel=3, output_channel=16, stride=1)   # N,16,256
        self.layer2 = ResBlock(input_channel=16, output_channel=32, stride=2)  # N,32,128
        self.layer3 = ResBlock(input_channel=32, output_channel=64, stride=2)  # N,64,64
        self.layer4 = ResBlock(input_channel=64, output_channel=96, stride=2)  # N,96,32
        self.layer5 = ResBlock(input_channel=96, output_channel=128, stride=2) # N,128,16
        self.layer6 = ResBlock(input_channel=128, output_channel=256, stride=2) # N,128,8
        self.layer7 = ResBlock(input_channel=256, output_channel=128, stride=2) # N,128,4
        self.layer8 = ResBlock(input_channel=128, output_channel=64, stride=1) # N,128,4
        # self.layer9 = ResBlock(input_channel=64, output_channel=32, stride=1) # N,32,4
        # self.layer10 = ResBlock(input_channel=32, output_channel=16, stride=1) # N,16,4
        # Transformer 部分
        self.d_model = 64  # 嵌入维度，需要与输入特征维度一致
        self.nhead = 4
        self.num_layers = 1
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=500)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        # 全连接层部分
        self.fc1 = nn.Linear(256, 256)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Sequential(
            nn.Linear(256, config.out_feature),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Linear(config.out_feature, 1)

    def forward(self, x):
        '''
        :param x: shape:(N, 3, 128)
        :return:
        '''
        # CNN 特征提取
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)  # 输出形状: [N, 128, 8]
        out = self.layer6(out)  # 输出形状: [N, 128, 8]
        out = self.layer7(out)  # 输出形状: [N, 128, 8]
        out = self.layer8(out)  # 输出形状: [N, 128, 4]
        # out = self.layer9(out)  # 输出形状: [N, 128, 8]
        # out = self.layer10(out)  # 输出形状: [N, 128, 8]
        # 调整形状以适配 Transformer 输入
        out = out.permute(0, 2, 1)  # [N, 4, 128]
        # 添加位置编码
        out = self.pos_encoder(out)  # [N, 8, 128]
        # Transformer 编码器
        out = self.transformer_encoder(out)  # [N, 8, 128]
        # 展平
        out = out.reshape(out.size(0), -1)  # [N, 8 * 128]
        # 全连接层
        out = self.fc1(out)
        feature = self.relu(out)
        feature = self.fc2(feature)
        pred = self.fc3(feature)
        # pred = self.predictor(out)
        return feature, pred



class PatchEmbedding(nn.Module):
    def __init__(self, length, patch_size, in_channels, embedded_dim, dropout=0, norm_layer=None):
        super(PatchEmbedding, self).__init__()
        self.Patch_embedding = nn.Conv1d(in_channels=in_channels, out_channels=embedded_dim, kernel_size=patch_size,
                                         stride=patch_size)
        self.Norm = norm_layer(embedded_dim) if norm_layer else nn.Identity()
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.Patch_embedding(x)  # 对数据进行分块、编码  (batch, 1, 样本长度) --> (batch, 编码维度, 块的个数)
        x = x.transpose(2, 1)  # 调整维度  (batch, 编码维度, 块的个数) --> (batch, 块的个数, 编码维度)
        x = self.Norm(x)
        x = self.Dropout(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的维度
                 num_heads=8,  # 多头自注意力机制 头数
                 qkv_bias=False,  # 生成QKV时是否使用偏执，默认False
                 qk_norm=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_layer=nn.LayerNorm, ):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x:[batch_size, num_patches+1, total_embed_dim]
        B, N, C = x.shape
        # qkv():-->[batch_size, num_patches+1, 3*total_embed_dim]
        # total_embed_dim = num_heads * embed_dim_per_head
        # reshape:-->[batch_size, num_patches+1, 3, num_heads, embed_dim_per_head]
        # permute:-->[3, batch_size, num_heads, num_patches+1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q,k,v形状：-->[batch_size, num_heads, num_patches+1, embed_dim_per_head]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        # transpose:-->[batch_size, num_heads, embed_dim_per_head, num_patches+1]
        # @ :-->[batch_size, num_heads, num_patches+1, num_patches+1]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # @ :-->[batch_size, num_heads, num_patches+1, embed_dim_per_head]
        # transpose:-->[batch_size, num_patches+1, num_heads, embed_dim_per_head]
        # reshape:-->[batch_size, num_patches+1, total_embed_dim]
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,  # 输入维度
                 hidden_features=None,  # 第一个全连接层神经元个数
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0., ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.kanliner_1 = KANLinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.kanliner_2 = KANLinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        # x= self.kanliner_1(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.kanliner_2(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim,  # 每个token的维度
                 num_heads,  # 多头自注意力机制 头数
                 mlp_ratio=4.,  # MLP层中第一个全连接层是输入维度的4倍
                 qkv_bias=False,
                 qk_norm=False,
                 proj_drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop,
                              proj_drop=proj_drop, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 网格大小，默认为 5
        spline_order=3,  # 分段多项式的阶数，默认为 3
        scale_noise=0.1,  # 缩放噪声，默认为 0.1
        scale_base=1.0,   # 基础缩放，默认为 1.0
        scale_spline=1.0,    # 分段多项式的缩放，默认为 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
        grid_eps=0.02,
        grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = ( # 生成网格
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # 初始化基础权重和分段多项式权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1 # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )





class CNN_Transformer_kan(nn.Module):
    '''
    input shape: (N, 3, 128)
    '''
    def __init__(self, config, in_c=8, norm_layer=nn.LayerNorm, qkv_bias=True, qk_norm=None,):
        super(CNN_Transformer_kan, self).__init__()
        # CNN 部分
        self.layer1 = ResBlock(input_channel=3, output_channel=16, stride=1)   # N,16,256
        self.layer2 = ResBlock(input_channel=16, output_channel=32, stride=2)  # N,32,128
        self.layer3 = ResBlock(input_channel=32, output_channel=64, stride=2)  # N,64,64
        self.layer4 = ResBlock(input_channel=64, output_channel=96, stride=2)  # N,96,32
        self.layer5 = ResBlock(input_channel=96, output_channel=128, stride=2) # N,128,16
        self.layer6 = ResBlock(input_channel=128, output_channel=256, stride=2) # N,256,8
        self.layer7 = ResBlock(input_channel=256, output_channel=128, stride=2) # N,128,4
        self.layer8 = ResBlock(input_channel=128, output_channel=64, stride=1) # N,64,4
        # self.layer9 = ResBlock(input_channel=64, output_channel=32, stride=1) # N,32,4
        # self.layer10 = ResBlock(input_channel=32, output_channel=16, stride=1) # N,16,4
        # Transformer 部分
        self.d_model = 256  # 嵌入维度，需要与输入特征维度一致
        self.nhead = 16
        self.num_layers = 4
        self.Cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.Patch_embedding = PatchEmbedding(config.L_sample, config.Patch_size, in_c, self.d_model)
        self.Blocks = nn.Sequential(*[
            Block(dim=self.d_model, num_heads=self.nhead, mlp_ratio=config.Mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, norm_layer=norm_layer) for i in range(self.num_layers)])
        self.Norm = norm_layer(self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=500)
        # encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        # 全连接层部分
        self.kanlinear_1 = KANLinear(256, config.out_feature)
        self.kanlinear_2 = KANLinear(config.out_feature, 1)
        self.fc1 = nn.Linear(64, 256)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Sequential(
            nn.Linear(256, config.out_feature),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Linear(config.out_feature, 1)

    def forward_encoder(self, x):
        x = self.Patch_embedding(x)
        Cls_tokens = self.Cls_token.expand(x.shape[0], -1, -1)
        x_ = torch.cat((Cls_tokens, x), dim=1)
        x_embed_pos = self.pos_encoder(x_)
        out = self.Blocks(x_embed_pos)
        return out[:, 0, :]

    def forward(self, x):
        '''
        :param x: shape:(N, 3, 128)
        :return:
        '''
        # CNN 特征提取
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)# 输出形状: [N, 256, 8]
        #out = self.layer7(out)
        #out = self.layer8(out)  # 输出形状: [N, 64, 4]

        # 调整形状以适配 Transformer 输入
        out = out.permute(0, 2, 1)  # [N, 8,256]


        #feature_2= feature_2.reshape(out.size(0), -1)  # [N, 8 * 128]
        # 添加位置编码
        out = self.forward_encoder(out)
        # 展平
        out = out.reshape(out.size(0), -1)  # [N, 4*64]
        # 全连接层
        feature = self.kanlinear_1(out)
        pred = self.kanlinear_2(feature)
        return feature, pred





class CNN_Transformer_kan_ADV(nn.Module):
    '''
    input shape: (N, 3, 128)
    '''
    def __init__(self, config, in_c=64, norm_layer=nn.LayerNorm, qkv_bias=True, qk_norm=None,):
        super(CNN_Transformer_kan_ADV, self).__init__()
        # CNN 部分
        self.layer1 = ResBlock(input_channel=3, output_channel=16, stride=1)   # N,16,256
        self.layer2 = ResBlock(input_channel=16, output_channel=32, stride=2)  # N,32,128
        self.layer3 = ResBlock(input_channel=32, output_channel=64, stride=2)  # N,64,64
        self.layer4 = ResBlock(input_channel=64, output_channel=96, stride=2)  # N,96,32
        self.layer5 = ResBlock(input_channel=96, output_channel=128, stride=2) # N,128,16
        self.layer6 = ResBlock(input_channel=128, output_channel=256, stride=2) # N,128,8
        self.layer7 = ResBlock(input_channel=256, output_channel=128, stride=2) # N,128,4
        self.layer8 = ResBlock(input_channel=128, output_channel=64, stride=1) # N,128,4
        # self.layer9 = ResBlock(input_channel=64, output_channel=32, stride=1) # N,32,4
        # self.layer10 = ResBlock(input_channel=32, output_channel=16, stride=1) # N,16,4
        # Transformer 部分
        self.d_model = 64  # 嵌入维度，需要与输入特征维度一致
        self.nhead = 1
        self.num_layers = 1
        self.Cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.Patch_embedding = PatchEmbedding(config.L_sample, config.Patch_size, in_c, self.d_model)
        self.Blocks = nn.Sequential(*[
            Block(dim=self.d_model, num_heads=config.N_heads, mlp_ratio=config.Mlp_ratio, qkv_bias=qkv_bias,
                  qk_norm=qk_norm, norm_layer=norm_layer) for i in range(config.Depth)])
        self.Norm = norm_layer(self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=500)
        # encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        # 全连接层部分
        self.kanlinear_1 = KANLinear(64, config.out_feature)
        self.kanlinear_2 = KANLinear(config.out_feature, 1)
        self.fc1 = nn.Linear(64, config.out_feature)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Sequential(
            nn.Linear(256, config.out_feature),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Linear(config.out_feature, 1)

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=config.out_feature, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=2)
        )
    def forward_encoder(self, x):
        x = self.Patch_embedding(x)
        Cls_tokens = self.Cls_token.expand(x.shape[0], -1, -1)
        x_ = torch.cat((Cls_tokens, x), dim=1)
        x_embed_pos = self.pos_encoder(x_)
        out = self.Blocks(x_embed_pos)
        return out[:, 0, :]

    def forward(self, x,alpha):
        '''
        :param x: shape:(N, 3, 128)
        :return:
        '''
        # CNN 特征提取
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)  # 输出形状: [N, 128, 8]
        # out = self.layer6(out)  # 输出形状: [N, 128, 8]
        # out = self.layer7(out)  # 输出形状: [N, 128, 8]
        # out = self.layer8(out)  # 输出形状: [N, 128, 8]
        # out = self.layer9(out)  # 输出形状: [N, 128, 8]
        # out = self.layer10(out)  # 输出形状: [N, 128, 8]
        # 调整形状以适配 Transformer 输入
        out = out.permute(0, 2, 1)  # [N, 8, 128]
        # 添加位置编码
        out = self.forward_encoder(out)
        # out = self.pos_encoder(out)  # [N, 8, 128]
        # # Transformer 编码器
        # out = self.transformer_encoder(out)  # [N, 8, 128]
        # 展平
        out = out.reshape(out.size(0), -1)  # [N, 8 * 128]
        # 全连接层
        feature = self.kanlinear_1(out)
        pred = self.kanlinear_2(feature)
        domain_outputs = self.domain_classifier(feature)
        # out = self.fc1(out)
        # feature = self.relu(out)
        # # feature = self.fc2(feature)
        # pred = self.fc3(feature)
        # pred = self.predictor(out)
        return feature, pred, domain_outputs

class MMDNet(nn.Module):
    def __init__(self, in_channel=3):
        super(MMDNet, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=10, kernel_size=11, padding=5, padding_mode='zeros'),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=10, stride=2)
        )
        self.feature2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=11, padding=5, padding_mode='zeros'),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=10, stride=2)
        )
        self.feature3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=11, padding=5, padding_mode='zeros'),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=10, stride=2)
        )
        self.feature4 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=11, padding=5, padding_mode='zeros'),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=10, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear( in_features=80,out_features=256),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, L_sample=256):
        #x = x.view(-1, 1, L_sample)
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        feature_flatten = self.feature4(x)
        feature_fc = self.classifier(feature_flatten)
        pred = self.fc2(feature_fc)

        return pred, feature_flatten, feature_fc


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Baseline(nn.Module):
    """
    纯 CNN baseline，接口与你的主模型一致： forward(x) -> (feature, pred)
    设计要点:
      - 使用与主模型类似的 ResBlock 堆栈（保证感受野、通道演进类似）
      - 最后用 AdaptiveAvgPool1d(1) -> flatten 得到固定维度的特征向量
      - feature_dim = config.out_feature（默认为 256）
    输入: x shape (N, in_c, L)  (例如 (N,3,128))
    输出:
      feature: tensor (N, config.out_feature)
      pred:    tensor (N, 1)
    """
    def __init__(self, config, in_c=3):
        super(CNN_Baseline, self).__init__()
        # 保持和主模型类似的残差块配置，但略微简化
        self.layer1 = ResBlock(input_channel=in_c, output_channel=16, stride=1)
        self.layer2 = ResBlock(input_channel=16, output_channel=32, stride=2)
        self.layer3 = ResBlock(input_channel=32, output_channel=64, stride=2)
        self.layer4 = ResBlock(input_channel=64, output_channel=96, stride=2)
        self.layer5 = ResBlock(input_channel=96, output_channel=128, stride=2)
        self.layer6 = ResBlock(input_channel=128, output_channel=256, stride=2)

        # 全局池化 -> 得到 (N, 256, 1) -> flatten -> (N, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接映射到 feature_dim (与 config.out_feature 对齐)
        hidden_dim = 256
        self.fc1 = nn.Linear(256, hidden_dim)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, config.out_feature)  # feature dim
        self.pred_head = nn.Linear(config.out_feature, 1)

        # 初始化（参考你文件风格）
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        nn.init.xavier_uniform_(self.pred_head.weight)

    def forward(self, x):
        # x: (N, in_c, L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)   # (N, 256, L')
        out = self.global_pool(out)  # (N, 256, 1)
        out = out.view(out.size(0), -1)  # (N, 256)
        out = self.act(self.fc1(out))
        feature = self.fc2(out)           # (N, config.out_feature)
        pred = self.pred_head(feature)    # (N,1)
        return feature, pred

class CNN_LSTM_Baseline(nn.Module):
    """
    CNN + LSTM 模型，同样返回 (feature, pred)
    设计要点:
      - 用 CNN（ResBlock）提取局部/time-frequency 特征
      - 将卷积输出按时间步序列化后输入 LSTM 捕捉长程依赖
      - 最终输出同样是 config.out_feature 维度的 feature 向量
    参数:
      config.out_feature: 最终 feature 维度（和其它模型统一）
      config.lstm_hidden (可选)：LSTM 隐状态维度，默认 128
      config.lstm_layers (可选)：LSTM 层数，默认 1
    """
    def __init__(self, config, in_c=3):
        super(CNN_LSTM_Baseline, self).__init__()
        self.layer1 = ResBlock(input_channel=in_c, output_channel=16, stride=1)
        self.layer2 = ResBlock(input_channel=16, output_channel=32, stride=2)
        self.layer3 = ResBlock(input_channel=32, output_channel=64, stride=2)
        self.layer4 = ResBlock(input_channel=64, output_channel=96, stride=2)
        self.layer5 = ResBlock(input_channel=96, output_channel=128, stride=2)
        # 在这里保留一个中等通道数以便送入 LSTM（channels -> feature per time step）
        # 不把通道扩展到 256，保持 LSTM 输入大小可控
        self.layer6 = ResBlock(input_channel=128, output_channel=128, stride=1)

        # LSTM 参数从 config 中读取（若无则给默认值）
        self.lstm_hidden = getattr(config, "lstm_hidden", 512)
        self.lstm_layers = getattr(config, "lstm_layers", 4)
        self.lstm = nn.LSTM(
            input_size=128,            # 来自 conv 输出的 channel 数
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # 全连接层将 LSTM 的最后时间步映射到 feature_dim
        self.fc1 = nn.Linear(self.lstm_hidden, 256)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, config.out_feature)   # feature dim
        self.pred_head = nn.Linear(config.out_feature, 1)

        # 初始化
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='linear')
        nn.init.xavier_uniform_(self.pred_head.weight)

    def forward(self, x):
        # x: (N, in_c, L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)   # (N, 128, L')
        # 准备 LSTM 输入： (N, L', channels) where channels=128
        out = out.permute(0, 2, 1)   # (N, L', 128)
        # LSTM 输出 (out_seq, (h_n, c_n)); out_seq shape = (N, L', hidden)
        out_seq, _ = self.lstm(out)
        last_step = out_seq[:, -1, :]   # (N, hidden)
        out = self.act(self.fc1(last_step))
        feature = self.fc2(out)         # (N, config.out_feature)
        pred = self.pred_head(feature)  # (N,1)
        return feature, pred