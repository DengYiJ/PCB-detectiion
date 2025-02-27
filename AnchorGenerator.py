import torch

class AnchorGenerator:
    def __init__(self, grid_size, scales, aspect_ratios):
        self.grid_size = grid_size  # 网格大小 (S, S)
        self.scales = scales  # 锚框尺度
        self.aspect_ratios = aspect_ratios  # 锚框长宽比

    def generate_anchors(self):
        # 生成网格中心点
        grid = torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size))
        grid_centers = torch.stack(grid, dim=-1).view(-1, 2)  # (S*S, 2)

        # 生成锚框
        anchors = []
        for scale in self.scales:
            for aspect_ratio in self.aspect_ratios:
                w = scale * torch.sqrt(1.0 / aspect_ratio)
                h = scale * torch.sqrt(aspect_ratio)
                anchors.append(torch.tensor([w, h]))

        anchors = torch.tensor(anchors).view(-1, 2)  # (A, 2)
        anchors = anchors + grid_centers[:, None, :]  # (S*S, A, 2)

        return anchors.view(-1, 4)  # (S*S*A, 4)

# 示例
grid_size = 10  # 网格大小
scales = [0.5, 1.0, 2.0]  # 锚框尺度
aspect_ratios = [0.5, 1.0, 2.0]  # 锚框长宽比

anchor_generator = AnchorGenerator(grid_size, scales, aspect_ratios)
anchors = anchor_generator.generate_anchors()
print("Number of anchors:", anchors.shape[0])  # 输出锚框数量