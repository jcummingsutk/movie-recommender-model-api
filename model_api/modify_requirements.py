import argparse

if __name__ == "__main__":
    # Reads the requirements file and then splits the torch requirements from non-torch requirements so that I can install the CPU version of PyTorch
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements-file")
    parser.add_argument("--torch-requirements-out-file")
    parser.add_argument("--non-torch-requirements-out-file")

    args = parser.parse_args()

    requirements_file = args.requirements_file
    torch_requirements_file = args.torch_requirements_out_file
    non_torch_requirements_file = args.non_torch_requirements_out_file

    torch_reqs = []
    non_torch_reqs = []
    with open(requirements_file, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            if "torch" in line:
                torch_reqs.append(line)
            else:
                non_torch_reqs.append(line)
    with open(torch_requirements_file, "w") as fp:
        for line in torch_reqs:
            fp.write(line)
    with open(non_torch_requirements_file, "w") as fp:
        for line in non_torch_reqs:
            fp.write(line)
