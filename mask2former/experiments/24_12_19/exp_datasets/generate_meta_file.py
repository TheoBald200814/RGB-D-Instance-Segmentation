from mask2former.tools.data_process import generate_meta_file, get_image_name_list
import os


def main():
    train_image_name_list = get_image_name_list(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/train/images")
    train_mask_name_list = get_image_name_list(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/train/mask")
    valid_image_name_list = get_image_name_list(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/valid/images")
    valid_mask_name_list = get_image_name_list(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/valid/mask")
    train_image_path_list = [os.path.join(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/train/images",
        image_name) for image_name in train_image_name_list]
    train_mask_path_list = [os.path.join(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/train/mask",
        mask_name) for mask_name in train_mask_name_list]
    valid_image_path_list = [os.path.join(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/valid/images",
        image_name) for image_name in valid_image_name_list]
    valid_mask_path_list = [os.path.join(
        "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets/valid/mask",
        mask_name) for mask_name in valid_mask_name_list]
    output_dir = "/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/24_12_19/exp_datasets"

    generate_meta_file(train_image_path_list, train_mask_path_list, valid_image_path_list, valid_mask_path_list,
                       output_dir)


if __name__ == "__main__":
    main()