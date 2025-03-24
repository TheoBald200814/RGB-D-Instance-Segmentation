import cv2
from mask2former.utils.data_process import cosine_similarity_fuse_v3, csf_viewer_v2

def csf_test():
    laplace_img = cv2.imread("/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense/laplace_depth/png/aauwuouqwq_59.png")
    ahe_img = cv2.imread("/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense/ahe_depth/png/aauwuouqwq_59.png")
    guassian_img = cv2.imread("/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense/gaussian_depth/png/aauwuouqwq_59.png")

    decimation_img = cv2.imread("/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense/decimation_depth/png/aauwuouqwq_59.png")
    rs_img = cv2.imread("/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense/depth_colormap_by_rs/png/aauwuouqwq_59.png")
    spatial_img = cv2.imread("/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense/spatial_depth/png/aauwuouqwq_59.png")
    hole_filling_img = cv2.imread("/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/dataset/local/processed_realsense/hole_filling_depth/png/aauwuouqwq_59.png")

    original_images_list = [decimation_img, rs_img, spatial_img, hole_filling_img]

    fused_image_result = cosine_similarity_fuse_v3(original_images_list, check=csf_viewer_v2)

if __name__ == '__main__':
    csf_test()