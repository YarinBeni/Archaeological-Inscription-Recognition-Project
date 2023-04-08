import CustomDataset as cd
import CustomModel as cm
import copy

import pytorch_lightning as pl
import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.transforms.dino_transform import DINOTransform

# Future Tasks:
# todo:1a) find relevant degradation for the model:
#  possible list of ideas:
#   invraint to: camera position,random cracks, old photo filter, random deletes(random crops),random lighting(normalization?)
#      1b) future test combination of best degradations as did in "Simclr" paper?
# todo:2) figure out how to train on gpu local or in googlecolab
# todo:3) understand how pytorch experiments and weights and baiss from FSDL Lab02 and lab04
# todo:4) start model training
# todo:5) make sure overfit on single batch
# todo:6) expand to 100 epochs and check results on maybe another models
if __name__ == '__main__':
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    custom_pytorch_dataset = cd.CustomDataset(cd.CSV_DATASET_PATH)
    model = cm.DINO()
    transform = DINOTransform()
    dataset = LightlyDataset.from_torch_dataset(custom_pytorch_dataset, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=False,
                                             num_workers=8
                                             )

    trainer = pl.Trainer(overfit_batches=1, max_epochs=10, devices=1, accelerator=accelerator, log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=dataloader)

    #
    # def generate_embeddings(model, dataloader):
    #     """Generates representations for all images in the dataloader with
    #     the given model
    #     """
    #
    #     embeddings = []
    #     filenames = []
    #     with torch.no_grad():
    #         for img, label, fnames in dataloader:
    #             img = img.to(model.device)
    #             emb = model.backbone(img).flatten(start_dim=1)
    #             embeddings.append(emb)
    #             filenames.extend(fnames)
    #
    #     embeddings = torch.cat(embeddings, 0)
    #     embeddings = normalize(embeddings)
    #     return embeddings, filenames
    #
    #
    # model.eval()
    # embeddings, filenames = generate_embeddings(model, dataloader_test)
    #
    #
    # def get_image_as_np_array(filename: str):
    #     """Returns an image as an numpy array"""
    #     img = Image.open(filename)
    #     return np.asarray(img)
    #
    #
    # def plot_knn_examples(embeddings, filenames, n_neighbors=3, num_examples=6):
    #     """Plots multiple rows of random images with their nearest neighbors"""
    #     # lets look at the nearest neighbors for some samples
    #     # we use the sklearn library
    #     nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    #     distances, indices = nbrs.kneighbors(embeddings)
    #
    #     # get 5 random samples
    #     samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)
    #
    #     # loop through our randomly picked samples
    #     for idx in samples_idx:
    #         fig = plt.figure()
    #         # loop through their nearest neighbors
    #         for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
    #             # add the subplot
    #             ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
    #             # get the correponding filename for the current index
    #             fname = os.path.join(path_to_data, filenames[neighbor_idx])
    #             # plot the image
    #             plt.imshow(get_image_as_np_array(fname))
    #             # set the title to the distance of the neighbor
    #             ax.set_title(f"d={distances[idx][plot_x_offset]:.3f}")
    #             # let's disable the axis
    #             plt.axis("off")
