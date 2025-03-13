from mask2former.tools.data_process import generate_meta_file, get_image_name_list
import os


def main():
    root_path = "dataset/local/experiment_tiny_set/"
    train_image_name_list = get_image_name_list(os.path.join(root_path, "train/images"))
    train_mask_name_list = get_image_name_list(os.path.join(root_path, "train/mask"))
    valid_image_name_list = get_image_name_list(os.path.join(root_path, "valid/images"))
    valid_mask_name_list = get_image_name_list(os.path.join(root_path, "valid/mask"))
    train_image_path_list = [os.path.join(os.path.join(root_path, "train/images"), image_name) for image_name in train_image_name_list]
    train_mask_path_list = [os.path.join(os.path.join(root_path, "train/mask"), mask_name) for mask_name in train_mask_name_list]
    valid_image_path_list = [os.path.join(os.path.join(root_path, "valid/images"), image_name) for image_name in valid_image_name_list]
    valid_mask_path_list = [os.path.join(os.path.join(root_path, "valid/mask"), mask_name) for mask_name in valid_mask_name_list]
    output_dir = root_path

    generate_meta_file(train_image_path_list, train_mask_path_list, valid_image_path_list, valid_mask_path_list,
                       output_dir)


if __name__ == "__main__":
    main()