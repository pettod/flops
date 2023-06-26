import torch.nn as nn
from config import CONFIG


INPUT_SHAPE = (1, 1, 256)
SINGLE_VALUE_OPERATIONS = ["ReLU", "Sigmoid", "Tanh"]
POOL1D_LAYERS = ["MaxPool1d", "AvgPool1d"]
CONV1D_LAYERS = ["Conv1d", "ConvTranspose1d"]
CONV2D_LAYERS = ["Conv2d", "ConvTranspose2d"]
LINEAR_LAYERS = ["Linear"]
IGNORED_LAYERS = []


def conv1dOperations(layer_name, w_i, h_i, k, p, s, b, c_i, c_o):
    # Output width and height
    border_size = 2 * ((k-1) // 2 - p)
    if layer_name == CONV1D_LAYERS[0] or layer_name in POOL1D_LAYERS:
        w_o = w_i // s - border_size
        h_o = h_i // s - border_size
    elif layer_name == CONV1D_LAYERS[1]:
        w_o = w_i * s + border_size
        h_o = h_i * s + border_size
    w_o = max(w_o, 1)
    h_o = max(h_o, 1)

    # Flops
    multiplications = w_o * h_o * c_o * k * c_i
    additions = w_o * h_o * c_o * (k * c_i - 1 + b)
    parameters = (k * c_i + b) * c_o

    return (
        multiplications,
        additions,
        parameters,
        w_o,
        h_o,
    )


def flops(model):
    layers = [
        module for module in model.modules() if type(module) != nn.Sequential]
    c_i, w_i, h_i = INPUT_SHAPE
    c_o, w_o, h_o = c_i, w_i, h_i
    total_additions = 0
    total_multiplications = 0
    total_single_value_operations = {}
    total_number_of_parameters = 0
    not_counted_layers = []

    # Loop every layer
    for layer in layers:
        layer_name = str(layer).split("(")[0]

        # 1D convolutions
        if layer_name in CONV1D_LAYERS:

            # Kernel, padding, stride, bias, input channels, output channels
            k = layer.kernel_size[0]
            p = layer.padding[0]
            s = layer.stride[0]
            b = 0 if layer.bias is None else 1
            c_i = layer.in_channels
            c_o = layer.out_channels

            # Compute ops
            (
                multiplications,
                additions,
                parameters,
                w_o,
                h_o,
            ) = conv1dOperations(layer_name, w_i, h_i, k, p, s, b, c_i, c_o)
            total_multiplications += multiplications
            total_additions += additions
            total_number_of_parameters += parameters

            # Update next layer input shapes
            c_i = c_o
            w_i = w_o
            h_i = h_o

        # 1D pooling
        elif layer_name in POOL1D_LAYERS:

            # Kernel, padding, stride, bias, input channels, output channels
            k = layer.kernel_size
            p = layer.padding
            s = layer.stride
            k = k[0] if type(k) == tuple else k
            p = p[0] if type(p) == tuple else p
            s = s[0] if type(s) == tuple else s
            b = 0

            # Compute ops
            (
                _,
                additions,
                _,
                w_o,
                h_o,
            ) = conv1dOperations(layer_name, w_i, h_i, k, p, s, b, c_i, c_o)
            total_additions += additions

            # Update next layer input shapes
            w_i = w_o
            h_i = h_o

        # 2D convolutions
        elif layer_name in CONV2D_LAYERS:

            # Kernel, padding, stride, bias, input channels, output channels
            k = layer.kernel_size[0]
            p = layer.padding[0]
            s = layer.stride[0]
            b = 0 if layer.bias is None else 1
            c_i = layer.in_channels
            c_o = layer.out_channels

            # Output width and height
            border_size = 2 * ((k-1) // 2 - p)
            if layer_name == CONV2D_LAYERS[0]:
                w_o = w_i // s - border_size
                h_o = h_i // s - border_size
            elif layer_name == CONV2D_LAYERS[1]:
                w_o = w_i * s + border_size
                h_o = h_i * s + border_size

            # Flops
            multiplications = w_o * h_o * c_o * k**2 * c_i
            additions = w_o * h_o * c_o * (k**2 * c_i - 1 + b)
            total_additions += additions
            total_multiplications += multiplications
            total_number_of_parameters += (k**2 * c_i + b) * c_o

            # Update next layer input shapes
            c_i = c_o
            w_i = w_o
            h_i = h_o

        # Linear
        elif layer_name in LINEAR_LAYERS:
            f_i = layer.in_features
            f_o = layer.out_features
            b = 0 if layer.bias is None else 1

            multiplications = f_i * f_o
            additions = b * f_o
            total_additions += additions
            total_multiplications += multiplications
            total_number_of_parameters += multiplications + additions

            # Update next layer input shapes
            c_i = 1
            w_i = f_o
            h_i = 1

        # Activations
        elif layer_name in SINGLE_VALUE_OPERATIONS:
            activations = w_i * h_i * c_i
            if layer_name in total_single_value_operations.keys():
                total_single_value_operations[layer_name] += activations
            else:
                total_single_value_operations[layer_name] = activations

        # Ignored layers
        elif layer_name in IGNORED_LAYERS:
            print(layer_name)
            continue

        # Not defined counting method for these layers
        else:
            not_counted_layers.append(layer_name)
    if len(not_counted_layers):
        print("Flops are not counted for the following layers:")
        for layer_name in not_counted_layers:
            print(layer_name)

    # Separately count flops
    flop_names = [
        "Additions",
        "Multiplications",
    ]
    flop_values = [
        total_additions,
        total_multiplications,
    ]
    for key, value in total_single_value_operations.items():
        flop_names.append(key)
        flop_values.append(value)
    total_number_of_flops = sum(flop_values)

    # Print
    printFlops(
        flop_names, flop_values, total_number_of_flops,
        total_number_of_parameters)
    return total_number_of_flops


def printFlops(
        flop_names, flop_values, total_number_of_flops,
        total_number_of_parameters):
    longest_value_length = 0
    for i in range(len(flop_values)):
        flop_values[i] = "{:,}".format(flop_values[i])
        if len(flop_values[i]) > longest_value_length:
            longest_value_length = len(flop_values[i])
    print(45 * "=")
    print("Flop types{}Flops".format(20 * " "))
    print(45 * "-")
    for i in range(len(flop_names)):
        name = flop_names[i]
        value = flop_values[i]
        number_of_spaces = \
            30 - len(name) + longest_value_length - len(value)
        print("{}{}{}".format(name, " " * number_of_spaces, value))
    print(45 * "=")
    texts  = [
        "Total number of flops",
        "Total number of parameters",
        "Total number of flops per pixel",
    ]
    text_numbers = [
        total_number_of_flops,
        total_number_of_parameters,
        int(total_number_of_flops / INPUT_SHAPE[1] / INPUT_SHAPE[2]),
    ]
    for i in range(len(texts)):
        text = texts[i]
        text_number = "{:,}".format(text_numbers[i])
        number_of_spaces = \
            30 - len(text) + longest_value_length - len(text_number)
        print("{}{}{}".format(text, " " * number_of_spaces, text_number))


def main():
    model = CONFIG.MODELS[0]
    flops(model)


if __name__ == "__main__":
    main()
