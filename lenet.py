# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace
from enum import Enum
import torch.fx as fx

# %%
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

# %%
model = LeNet5(num_classes=10) # 10 for MNIST digit recognition
x = torch.randn(1, 1, 32, 32)
y = model(x)
# print(y)

# %%
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced: torch.fx.GraphModule = symbolic_trace(model)

# High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced.graph)

# %%
symbolic_traced.graph.print_tabular()

# %%
# for node in symbolic_traced.graph.nodes:
#     if node.op == "call_module":
#         print(node.op, node.target, node.args, getattr(symbolic_traced, node.target))
#         print(type(getattr(symbolic_traced, node.target)))
#     elif node.op == "call_method":
#         print(node.op, node.target, node.args)

# %%
# Helper function that calculates the size of each layer in the ECG
def calculate_IRS_SIZE():
    traced = symbolic_trace(model)
    module_to_size = {}

    for node in traced.graph.nodes:
        if node.op == "call_module":
            module = getattr(traced, node.target)

            def make_hook(name):
                def hook(module, input, output):
                    module_to_size[name] = output.shape
                return hook

            module.register_forward_hook(make_hook(node.target))
    
    traced(x)
    return module_to_size

IRS_sizes = calculate_IRS_SIZE()
# print("IRS Sizes:", IRS_sizes)

# %%
all_operators = [node for node in symbolic_traced.graph.nodes if node.op == "call_module" or node.op == "call_function"]
# print(all_operators)

# %%
# for node in symbolic_traced.graph.nodes:
#     if node.op == "call_module":
#         print(node, list(node.users.keys()), node.all_input_nodes)

# %% [markdown]
# # Fusion Plan Algorithm

# %%
# Define an enum of values many-to-many, one-to-one
class mapping_type(Enum):
    MANY_TO_MANY = 1
    ONE_TO_ONE = 2
    SHUFFLE = 3
    REORGANIZE = 4

mapping_type_table = {
    torch.nn.modules.batchnorm.BatchNorm2d: mapping_type.ONE_TO_ONE,
    torch.nn.modules.activation.ReLU: mapping_type.ONE_TO_ONE,
    torch.nn.modules.pooling.MaxPool2d: mapping_type.MANY_TO_MANY,
    torch.nn.modules.conv.Conv2d: mapping_type.MANY_TO_MANY,
    torch.nn.modules.linear.Linear: mapping_type.SHUFFLE
}

# %%
# # Testing table lookup
# for node in symbolic_traced.graph.nodes:
#     if node.op == "call_module":
#         print(node.op, node.target, node.args, getattr(symbolic_traced, node.target))
#         print(mapping_type_table.get(type(getattr(symbolic_traced, node.target))))

# %%
class MAPPING_RELATIONSHIP(Enum):
    FUSE_BREAK = 0
    FUSE_ONE_TO_ONE = 1
    FUSE_MANY_TO_MANY = 2
    FUSE_SHUFFLE = 3

def mapping_check(op, successor):
    op1_mapping = mapping_type_table.get(type(getattr(symbolic_traced, op.target)))
    successor_mapping = mapping_type_table.get(type(getattr(symbolic_traced, successor.target)))

    # CASES depending on op and successor
    if op1_mapping == mapping_type.ONE_TO_ONE and successor_mapping == mapping_type.ONE_TO_ONE:
        return MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE
    if op1_mapping == mapping_type.MANY_TO_MANY and successor_mapping == mapping_type.MANY_TO_MANY:
        return MAPPING_RELATIONSHIP.FUSE_BREAK
    if op1_mapping == mapping_type.MANY_TO_MANY and successor_mapping == mapping_type.ONE_TO_ONE:
        return MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY
    if op1_mapping == mapping_type.ONE_TO_ONE and successor_mapping == mapping_type.MANY_TO_MANY:
        return MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY
    if op1_mapping == mapping_type.SHUFFLE and successor_mapping == mapping_type.SHUFFLE:
        return MAPPING_RELATIONSHIP.FUSE_SHUFFLE
    if op1_mapping == mapping_type.SHUFFLE and successor_mapping == mapping_type.ONE_TO_ONE:
        return MAPPING_RELATIONSHIP.FUSE_SHUFFLE
    if op1_mapping == mapping_type.SHUFFLE and successor_mapping == mapping_type.MANY_TO_MANY:
        return MAPPING_RELATIONSHIP.FUSE_BREAK # TODO: this should be a fuse check
    if op1_mapping == mapping_type.ONE_TO_ONE and successor_mapping == mapping_type.SHUFFLE:
        return MAPPING_RELATIONSHIP.FUSE_SHUFFLE
    if op1_mapping == mapping_type.MANY_TO_MANY and successor_mapping == mapping_type.SHUFFLE:
        return MAPPING_RELATIONSHIP.FUSE_BREAK # TODO: this should be a fuse check
    
    return MAPPING_RELATIONSHIP.FUSE_BREAK # DEFAULT CASE
    

def mapping_check_relationship(op_mapping, successor_mapping):
    # CASES depending on op and successor
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE and successor_mapping == MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE:
        return MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY and successor_mapping == MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY:
        return MAPPING_RELATIONSHIP.FUSE_BREAK
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY and successor_mapping == MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE:
        return MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE and successor_mapping == MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY:
        return MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_SHUFFLE and successor_mapping == MAPPING_RELATIONSHIP.FUSE_SHUFFLE:
        return MAPPING_RELATIONSHIP.FUSE_SHUFFLE
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_SHUFFLE and successor_mapping == MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE:
        return MAPPING_RELATIONSHIP.FUSE_SHUFFLE
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_SHUFFLE and successor_mapping == MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY:
        return MAPPING_RELATIONSHIP.FUSE_BREAK # TODO: this should be a fuse check
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_ONE_TO_ONE and successor_mapping == MAPPING_RELATIONSHIP.FUSE_SHUFFLE:
        return MAPPING_RELATIONSHIP.FUSE_SHUFFLE
    if op_mapping == MAPPING_RELATIONSHIP.FUSE_MANY_TO_MANY and successor_mapping == MAPPING_RELATIONSHIP.FUSE_SHUFFLE:
        return MAPPING_RELATIONSHIP.FUSE_BREAK # TODO: this should be a fuse check
    
    return MAPPING_RELATIONSHIP.FUSE_BREAK # DEFAULT CASE

# %%
def generate_seed(nodes):
    # Find all the one-to-one mapping operators
    one_to_one_nodes = [node for node in nodes if mapping_type_table.get(type(getattr(symbolic_traced, node.target))) == mapping_type.ONE_TO_ONE]
    # Using the IRS_size, return the one-to-one mapping operator with the smallest output size (take product of dimensions)
    min_size = float('inf')
    seed_node = None
    for node in one_to_one_nodes:
        size = np.prod(IRS_sizes[node.target])
        if size < min_size:
            min_size = size
            seed_node = node
    return seed_node
    

def successors(op):
    successors = list(op.users.keys())

    # Remove successors that meet the conditions
    successors = [s for s in successors if not (s.op == "output" or s.op == "call_method")]

    return successors

def predecessors(op):
    pred = list(op.all_input_nodes)

    # Check if the pred is "placeholder" or "size"/"reshape"
    pred = [p for p in pred if not (p.op == "call_method" or p.op == "placeholder")]

    return pred

def fuse_successor(op, successor, block):
    # Check the mapping relationship
    relation = mapping_check(op, successor)

    # no relationship exists
    if relation == MAPPING_RELATIONSHIP.FUSE_BREAK:
        return
    
    # Step 2.2: check constraint requirement

    # Step 2.3: if benefit of fusion unknown, get the latency and check 
    # TODO: this only applies if the fusion is potentially beneficial

    # Block is the combination of op and successor
    block.add(successor)
    # Step 2.4: Recursively head to successor
    for fusing_op in successors(successor):
        fuse_successor(successor, fusing_op, block)

def fuse_predecessor(sp, predecessor, block):
    # Check relation
    relation = mapping_check(sp, predecessor)

    # no relationship exists
    if relation == MAPPING_RELATIONSHIP.FUSE_BREAK:
        return
    
    # Step 2.2: check constraint requirement

    # Step 2.3: if benefit of fusion unknown, get the latency and check
    # TODO: this only applies if the fusion is potentially beneficial

    # Block is the combination of op and successor
    block.add(predecessor)

    # Step 2.4: Recursively head to predecessor
    for fusing_op in predecessors(predecessor):
        fuse_predecessor(sp, fusing_op, block)

# <Algorithm Entry>
unfused_ops = set(all_operators)
print("Unfused ops: ", unfused_ops)

fused_groups = dict()

# Step 1: start fuse from the selected seed
while (sp := generate_seed(unfused_ops)):
    print("Seed node: ", sp)
    block = {sp}
    # Step 2: head to successor
    for successor in successors(sp):
        fuse_successor(sp, successor, block)
    # Step 3: head to predecessors
    for predecessor in predecessors(sp):
        fuse_predecessor(sp, predecessor, block)
    unfused_ops -= block
        
    print("Fused ops: ", block)
    print("Unfused ops: ", unfused_ops)

    # Add block to fused_groups
    fused_groups[sp] = block

print("\n--- Fusion plan result ---")
print("Fused groups: ", fused_groups)

# %%
# Check for the best operators to fuse from each group of fused_groups
def find_best_fusion(nodes, seed_node):
    # from the seed node, check if the successor and predecssor are in the nodes list
    successor = successors(seed_node)[0]
    predecessor = predecessors(seed_node)[0]

    pred_fuse = [seed_node]
    suc_fuse = [seed_node]

    cur_suc_type = mapping_type_table.get(type(getattr(symbolic_traced, seed_node.target)))
    cur_pred_type = mapping_type_table.get(type(getattr(symbolic_traced, seed_node.target)))
    cur_suc_seed = seed_node
    cur_pred_seed = seed_node

    # in a loop, get all the successors
    while True:
        if successor in nodes:
            # Check if they can be fused
            fuse_result_type= mapping_check(cur_suc_seed, successor)
            if fuse_result_type == MAPPING_RELATIONSHIP.FUSE_BREAK:
                break
            else:
                suc_fuse.append(successor)
                cur_suc_type = fuse_result_type
                cur_suc_seed = successor
                successor = successors(successor)[0] if successors(successor) else None
        else:
            break

    while True:
        if predecessor in nodes:
            # Check if they can be fused
            fuse_result_type = mapping_check(cur_pred_seed, predecessor)
            if fuse_result_type == MAPPING_RELATIONSHIP.FUSE_BREAK:
                break
            else:
                pred_fuse.append(predecessor)
                cur_pred_type = fuse_result_type
                cur_pred_seed = predecessor
                predecessor = predecessors(predecessor)[0] if predecessors(predecessor) else None
        else:
            break

    # See if predecessor and successor can be fused
    suc_pred_fuse_type = mapping_check_relationship(cur_suc_type, cur_pred_type)

    if suc_pred_fuse_type == MAPPING_RELATIONSHIP.FUSE_BREAK:
        # pick the best one (number of fusions)
        if len(pred_fuse) > len(suc_fuse):
            return pred_fuse, suc_fuse[1:]
        else:
            return suc_fuse, pred_fuse[1:]
    else:
        return pred_fuse + suc_fuse[1:], []


def check_successor(nodes, successor):
    # Check if the successor is in the nodes list
    if successor in nodes:
        return True
    return False

def check_predecessor(nodes, predecessor):
    # Check if the predecessor is in the nodes list
    if predecessor in nodes:
        return True
    return False

operators_to_fuse = []
operators_unfused = []

for seed, seed_block in fused_groups.items():
    fuse, not_fused = find_best_fusion(seed_block, seed)
    operators_to_fuse.append(fuse)
    operators_unfused.extend(not_fused)

print("\n--- Fused operators ---")
print("Operators to fuse: ", operators_to_fuse)
print("Operator candidates that could not be fused: ", operators_unfused)
 

# %%


# %%


# %%


# %%


# %%
# # Dictionary to store output sizes
# layer_outputs = {}

# def hook_fn(module, input, output):
#     layer_outputs[module] = output.shape

# # Register hooks on all modules
# hooks = []
# for name, module in model.named_modules():
#     if not isinstance(module, nn.Sequential) and not isinstance(module, LeNet5):
#         hooks.append(module.register_forward_hook(hook_fn))

# # Create a dummy input and pass through model
# dummy_input = torch.randn(1, 1, 32, 32)
# _ = model(dummy_input)

# # Print the recorded output shapes
# for layer, shape in layer_outputs.items():
#     print(f"{layer.__class__.__name__} -> {shape}")

# # Remove the hooks (clean up)
# for hook in hooks:
#     hook.remove()


# %%
# # Print the recorded output shapes
# for layer, shape in layer_outputs.items():
#     print(f"{layer.__class__.__name__} -> {shape}")

# # Remove the hooks (clean up)
# for hook in hooks:
#     hook.remove()


# %%
# traced = symbolic_trace(model)
# module_to_size = {}

# for node in traced.graph.nodes:
#     if node.op == "call_module":
#         module = getattr(traced, node.target)

#         def make_hook(name):
#             def hook(module, input, output):
#                 module_to_size[name] = output.shape
#             return hook

#         module.register_forward_hook(make_hook(node.target))

# x = torch.randn(1, 1, 32, 32)
# traced(x)

# print("Result:", module_to_size)

# %%



