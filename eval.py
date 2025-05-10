import os
# os.environ["TORCH_COMPILE_DEBUG"] = "1"
import glob
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = LeNet5(num_classes=10)

compiled_model = torch.compile(model, mode="max-autotune")

# store graph IRs
with torch.no_grad():
    x = torch.randn(1, 1, 32, 32)
    y = compiled_model(x)

print(y)

# # Find most recent torch_compile_logs directory
# compile_logs_dir = "torch_compile_debug"
# latest_dir = max(glob.glob(os.path.join(compile_logs_dir, "*")), key=os.path.getmtime)
# print(latest_dir)

# # Find ir_post_fusion.txt recursively
# ir_post_fusion_path = ""
# for root, dirs, files in os.walk(latest_dir):
#     print(root, dirs, files)
#     if "ir_post_fusion.txt" in files:
#         ir_post_fusion_path = os.path.join(root, "ir_post_fusion.txt")
#         break

# # Read contents
# with open(ir_post_fusion_path, 'r') as f:
#     scheduler_output = f.read()

# # Regular expression to match operation names (op1, op2_op3, etc.)
# op_names = re.findall(r'\bop[0-9_]+\b', scheduler_output)

# # Remove duplicates and sort the list for a clean output
# unique_op_names = sorted(set(op_names))

# print(unique_op_names)