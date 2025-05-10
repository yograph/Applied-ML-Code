"""
This is to train the YOLOX
"""
KAGGLE = True


class train_yolox:
    """
    THis is to train a yolox
    
    """
    OUTPUT_RECT_DIR  = "../Cancer_Detection/data/dataset_images/rsna-mammography-images-as-pngs/images_as_pngs_1024/train_images_processed_1024/*/*"
    OUTPUT_FINAL_DIR = "../Cancer_Detection/data/dataset_images/rsna-mammography-images-as-pngs/images_as_pngs_1024/train_images_processed_1024/*/*"

    def __init__(self):
        if KAGGLE:
            pattern = "../input/rsna-mammography-images-as-pngs/images_as_pngs_1024/train_images_processed_1024/*/*"
        else:
            pattern = "../Cancer_Detection/data/dataset_images/rsna-mammography-images-as-pngs/images_as_pngs_1024/train_images_processed_1024/*/*"
        main = []



# class 