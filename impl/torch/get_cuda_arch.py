import torch


def get_cuda_architecture():
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(current_device)
        return "{}.{}".format(device_properties.major, device_properties.minor)
    else:
        return "N/A"


if __name__ == "__main__":
    cuda_architecture = get_cuda_architecture()
    print(cuda_architecture)
