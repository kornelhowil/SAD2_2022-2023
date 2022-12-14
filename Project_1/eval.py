import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import torch
import scanpy as sc
import os


def read_adata():
    train_path = 'data/SAD2022Z_Project1_GEX_train.h5ad'
    test_path = 'data/SAD2022Z_Project1_GEX_test.h5ad'
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)
    return train_adata, test_adata


def first_histplot(train_adata):
    preprocessed = train_adata.X
    raw = train_adata.layers['counts']

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    bins = np.arange(0, 10, 1)

    counts, bins = np.histogram(raw.toarray(), bins=bins)
    ax1.set_ylim([0, 3.5e8])
    ax1.hist(bins[:-1], bins, weights=counts)
    ax1.set_title('Histogram of the raw data')
    ax1.set(xlabel='Value', ylabel='Count')

    counts, bins = np.histogram(preprocessed.toarray(), bins=bins)
    ax2.set_ylim([0, 3.5e8])
    ax2.hist(bins[:-1], bins, weights=counts)
    ax2.set_title('Histogram of the preprocessed data')
    ax2.set(xlabel='Value', ylabel='Count')
    plt.clf()

    plt.savefig('Plots/histogram_full.pdf')


def second_histplot(train_adata):
    preprocessed = train_adata.X
    raw = train_adata.layers['counts']
    preprocessed_nz = preprocessed.toarray()[preprocessed.toarray() != 0]
    raw_nz = raw.toarray()[raw.toarray() != 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    bins = np.arange(0, 15, 1)

    counts, bins = np.histogram(raw_nz, bins=bins)
    ax1.set_ylim([0, 2.1e7])
    ax1.hist(bins[:-1], bins, weights=counts)
    ax1.set_title('Histogram of the raw data')
    ax1.set(xlabel='Value', ylabel='Count')

    counts, bins = np.histogram(preprocessed_nz, bins=bins)
    ax2.set_ylim([0, 2.1e7])
    ax2.hist(bins[:-1], bins, weights=counts)
    ax2.set_title('Histogram of the preprocessed data')
    ax2.set(xlabel='Value', ylabel='Count')

    plt.savefig('Plots/histogram_no_zeros.pdf')
    plt.clf()


def plot_pca(pca_var, test_adata, name, hue):
    sns.scatterplot(x=pca_var[:, 0], y=pca_var[:, 1], hue=test_adata.obs[hue], legend=False)
    plt.xlabel('First PCA component')
    plt.ylabel('Second PCA component')
    plt.savefig(f'Plots/{name}_PCA_{hue}.pdf')
    plt.clf()


def plot_model(name, test_adata):
    test_data = torch.Tensor(test_adata.layers['counts'].toarray())
    df = pd.read_csv(f'Models/{name}_losses.csv')

    plt.plot(df['train_loss'], label='Train')
    plt.plot(df['test_loss'], label='Test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('- ELBO')
    plt.savefig(f'Plots/{name}.pdf')
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    ax1.plot(df['train_kl_loss'], label='train')
    ax1.plot(df['test_kl_loss'], label='test')
    ax1.set(xlabel='Epoch', ylabel='Regularization loss')
    ax1.legend()
    ax2.plot(df['train_recon_loss'], label='train')
    ax2.plot(df['test_recon_loss'], label='test')
    ax2.set(xlabel='Epoch', ylabel='Reconstruction loss')
    ax2.legend()
    plt.savefig(f'Plots/{name}_losses.pdf')
    plt.clf()

    model = torch.load(f'Models/{name}_model.pt')
    model.eval()

    z = model.calc_latent(test_data)
    pca = PCA()
    pca.fit(z)
    variance = pca.explained_variance_ratio_
    combined_variance = np.zeros_like(variance)
    combined_variance[0] = variance[0]
    for i in range(1, len(variance)):
        combined_variance[i] = combined_variance[i - 1] + variance[i]
    plt.xlabel('Number of PCA components')
    plt.ylabel('Ratio of the explained variance')
    plt.axhline(y=0.95, color='black', linestyle='dashed', label='95% variance')
    plt.plot(combined_variance, label='Ratio of the explained variance')
    plt.legend()
    plt.savefig(f'Plots/{name}_PCA_components.pdf')

    pca_var = pca.fit_transform(z.cpu())
    plot_pca(pca_var, test_adata, name, 'cell_type')
    plot_pca(pca_var, test_adata, name, 'batch')
    plot_pca(pca_var, test_adata, name, 'DonorID')
    plot_pca(pca_var, test_adata, name, 'Site')


def main():
    if not os.path.exists('Plots'):
        os.mkdir('Plots')

    train_adata, test_adata = read_adata()
    print(train_adata)
    print(test_adata)

    first_histplot(train_adata)
    second_histplot(train_adata)
    plot_model('VanillaVAE50', test_adata)
    plot_model('VanillaVAE10', test_adata)
    plot_model('VanillaVAE5', test_adata)
    plot_model('CustomVAE10', test_adata)

    print("Figures are saved in the Plots directory.")


if __name__ == '__main__':
    main()
