import json 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
def visualize_and_save(data_path,base_image_path,significant_slice_path,mode="train"):
    print("Mode:", mode)
    df = pd.read_csv(significant_slice_path)
    result = (
    df.dropna(subset=["Slide"])       
      .groupby("Patient ID")
     .apply(lambda g: list(zip(g["Slide"], g["Bbox coordinates normalized (X, Y, W, H)"])))
      .apply(list)
      .to_dict()
)
    # all_process_files = ["OAS1_0002.json"]
    all_process_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    save_folder=f"viz_data/{mode}"
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    print("All process files:", all_process_files)
    for file in all_process_files:
        try:
            print("Processing file:", file)
            with open(os.path.join(data_path, file), "r") as f:
                processed_data = json.load(f)
            n_cols=8
            n_rows=8
            # dpi=100
            dpi=200
            # cell_px=128
            cell_px=256
            figsize=(
    n_cols * cell_px / dpi, n_rows * cell_px / dpi
            )
            for i,sample in enumerate(processed_data):
                save_path=os.path.join(save_folder, f"{sample['Patient ID']}_{i}_slices.png")
                print("Saving visualization to:", save_path)
                p_id= sample["Patient ID"]
                print("Processing Patient ID:", p_id)
                slice_order= sample["slice order"]
                print("result pid",result.get(p_id, []))
                special_idx=[(slice_order.index(s)) 
                                        for s, b in result[p_id]
                                                if s in slice_order]
                n=len(slice_order)
                fig, axes = plt.subplots(   
                    n_rows, n_cols,
                    figsize=figsize,
                    dpi=dpi
                )
                dementia_level = sample["A4"]
                overall_title = f"Patient ID: {p_id} | Dementia Level: {dementia_level}"
                fig.suptitle(overall_title, fontsize=12)
                axes = np.array(axes).reshape(-1)
                for i, ax in enumerate(axes):
                    if i >= n:
                        ax.axis("off")
                        continue
                    image_path = os.path.join(base_image_path, p_id, f"{slice_order[i]}.pkl")
                    with open(image_path, "rb") as f:
                        image = pd.read_pickle(f)
                    image=image.squeeze()
                    ax.imshow(image, cmap="gray")
                    ax.set_title(slice_order[i], fontsize=8, pad=4)
                    ax.axis("off")

                    if i in special_idx:
                        ax.add_patch(
                            plt.Rectangle(
                                (0, 0), 1, 1,
                                transform=ax.transAxes,
                                fill=False,
                                linewidth=2,
                                edgecolor="red"
                            )
                        )

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close(fig)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
        # return


def vizualize_significant_slices():
    # df=pd.read_csv(significant_slice_path)
    # print("number of significant slices:", len(df))
    # for index, row in df.iterrows():
    #     p_id=row["Patient ID"]
    #     slice_num=row["Slide"]
    #     image=row
    # pass         
    image_path="clean_data_s_chain/train/image/OAS1_0002/mpr-1_160.pkl"
    with open(image_path, "rb") as f:
        image = pd.read_pickle(f)
    save_path="test_slice.png"
    image=image.squeeze()
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    ) 

if __name__ == "__main__":
    # data_path="pseudo_3d/-1_all_slices/18eb3e36-0b00-43e8-b2ac-659ac19a94f6"
    # train_path= data_path + "/train/data/"
    # test_path= data_path + "/test/data/"
    # # train_image_path="clean_data_s_chain/train/image"
    # # test_image_path="clean_data_s_chain/test/image"
    # train_image_path="clean_data_s_chain/train/image_with_bboxes"
    # test_image_path="clean_data_s_chain/test/image_with_bboxes"
    # significant_slice_path="significant_slice.csv"
    # visualize_and_save(train_path,train_image_path,significant_slice_path,mode="train")
    # visualize_and_save(test_path,test_image_path,significant_slice_path,mode="test")
    vizualize_significant_slices()
    pass