import torch.nn as nn


def build_module_tree(module: nn.Module, module_name: str = "") -> dict:
    """Recursively traverse the given nn.Module and build a dictionary with."""
    # 1) Count direct parameters in this module
    direct_param_count = 0
    param_details = {}
    for pname, p in module.named_parameters(recurse=False):
        nump = p.numel()
        param_details[pname] = nump
        direct_param_count += nump

    # 2) Recursively process child modules
    children_info = {}
    for cname, child in module.named_children():
        children_info[cname] = build_module_tree(child, cname)

    # 3) Calculate total parameter count for this module (including all children)
    total = direct_param_count + sum(child["total"] for child in children_info.values())

    return {
        "name": module_name,
        "params": param_details,
        "children": children_info,
        "total": total,
    }


def print_module_tree(info: dict, parent_path: str = "", indent: int = 0):
    """
    Print the module tree built by build_module_tree() in a hierarchical format:
    "(total_parameter_count) (path_to_module_or_param)"
    The function sorts parameters and submodules in descending order of total size.
    """
    # Construct the current path
    name = info["name"]
    if not parent_path:
        full_path = name  # top level
    else:
        if name:  # submodule name is not empty
            full_path = f"{parent_path}/{name}"
        else:
            full_path = parent_path

    # Print total parameter count for the current module
    line = f"{info['total']:11,d} {full_path}"
    print(" " * indent + line)

    # Create a combined list of param_nodes (parameters) and child_nodes (submodules)
    param_nodes = []
    for param_name, count in info["params"].items():
        param_nodes.append(
            {
                "name": param_name,
                "params": {},
                "children": {},
                "total": count,
            }
        )

    child_nodes = list(info["children"].values())

    # Sort by 'total' in descending order
    combined = param_nodes + child_nodes
    combined.sort(key=lambda x: x["total"], reverse=True)

    # Recursively print all children
    for child_info in combined:
        print_module_tree(child_info, full_path, indent + 2)


def print_param_stats(model):
    """
    Prints formatted statistical information of the parameter values (not gradients)
    for the trainable parameters (.requires_grad=True) of the specified PyTorch model.

    - mean
    - std  (population standard deviation: std(unbiased=False))
    - L2 norm (param.data.norm())
    - RMS (root mean square: sqrt(mean(tensor^2)))

    The hierarchical name is displayed by replacing '.' with '/' in the default names
    (e.g., converting "layer.weight" to "layer/weight").
    """

    # List to temporarily store the statistics
    stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            mean_val = data.mean().item()
            std_val = data.std(unbiased=False).item()
            l2_val = data.norm().item()
            rms_val = data.pow(2).mean().sqrt().item()

            hierarchical_name = name.replace(".", "/")
            stats.append((hierarchical_name, mean_val, std_val, l2_val, rms_val))

    # Format function to display numbers in scientific notation with 3 significant digits
    def fmt(v):
        return f"{v:.3e}"

    # Column width settings (adjust if necessary)
    col_widths = [60, 15, 15, 15, 15]
    header_format = f"{{:<{col_widths[0]}}}{{:>{col_widths[1]}}}{{:>{col_widths[2]}}}{{:>{col_widths[3]}}}{{:>{col_widths[4]}}}"
    row_format = header_format

    # Print the header
    print(header_format.format("Parameter", "Mean", "Std", "L2 norm", "RMS"))
    print("-" * (sum(col_widths) + 1))

    # Print the main content
    for hname, mean_val, std_val, l2_val, rms_val in stats:
        print(
            row_format.format(
                hname,
                fmt(mean_val),
                fmt(std_val),
                fmt(l2_val),
                fmt(rms_val),
            )
        )
