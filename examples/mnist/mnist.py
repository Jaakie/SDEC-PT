import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset, Subset
import random
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
import uuid

from sdecpt.sdec import DEC
from sdecpt.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from sdecpt.utils import cluster_accuracy


class CachedMNIST(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = MNIST("./data", download=True, train=train, transform=img_transform)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return (
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()
            * 0.02
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = torch.tensor(
                    self._cache[index][1], dtype=torch.long
                ).cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)


@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=100,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=300,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    ds_train = CachedMNIST(
        train=True, cuda=cuda, testing_mode=testing_mode
    )  # training dataset
    
    ds_val = CachedMNIST(
        train=False, cuda=cuda, testing_mode=testing_mode
    )  # evaluation dataset
    
    #Create random indexes for semi-supervised learning. We make constraints for 20% of the data points.
    randidxs = []
    ratio = 0.2
    remainder = len(ds_train)%batch_size
    for i in range(int(len(ds_train)/batch_size)):
        randidx = []
        for j in range(int(batch_size**2 * ratio)):
            index1 = random.randrange(0, batch_size - 1)
            index2 = random.randrange(0, batch_size - 1)
            randidx.append((index1,index2))
        randidxs.append(np.array(randidx))
    if remainder!=0:
        randidx = []
        for j in range(int(remainder * ratio)):
            index1 = random.randrange(0, remainder - 1)
            index2 = random.randrange(0, remainder - 1)
            randidx.append((index1,index2))
        randidxs.append(np.array(randidx))
    
    autoencoder = StackedDenoisingAutoEncoder(
        [28 * 28, 500, 500, 2000, 10], final_activation=None
    )
    if cuda:
        autoencoder.cuda()

    #Debug - to load autoencoder or train
    loadAE = True

    if loadAE: 
        autoencoder = (torch.load('AE_model.pth'))
    else:
        print("Pretraining stage.")
        ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
        )
        print("Training stage.")
        ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
        ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback,
        )

        torch.save(autoencoder, 'AE_model.pth')
    print("DEC stage.")
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum = 0.9)

    #Debug - to load DEC model or train
    loadDEC = False

    if loadDEC:
        model = torch.load('DEC_model.pth')
    else:
        train(
        dataset=ds_train,
        idxes=randidxs,
        lamb=1e-4,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
        )

        torch.save(model, 'DEC_model.pth')

    predicted, actual, _, hidden_features = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
 
    print("Final DEC accuracy: %s" % accuracy)
    
    cluster_list=[]
    for i in range(10):
        cluster_list.append((actual==i))
    
    X_embedded = TSNE(n_components=2,learning_rate='auto', init='random').fit_transform(hidden_features.cpu().numpy())
    for i in range(10):
        plt.scatter(X_embedded[cluster_list[i],0],X_embedded[cluster_list[i],1])
    plt.title('Latent dimension from SDEC')
    plt.show()
    
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig(
            "confusion_%s.png" % confusion_id
        )
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()


if __name__ == "__main__":
    main()
