<div align="center">
  <h1>ESRGAN: Enhanced Super-Resolution Generative Adversarial Network</h1>

  <p>
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge" alt="License">
  </p>
</div>

<p>A complete, from-scratch PyTorch implementation of the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN). This model upscales low-resolution images by a scale factor of 4x, generating hyper-realistic textures and sharp edges by leveraging Residual-in-Residual Dense Blocks (RRDB) and a complex perceptual loss function.</p>

<hr>

<h2>🌟 Visual Results</h2>

<p>Below are the results of the model reconstructing high-resolution details from heavily degraded 64x64 inputs.</p>

<table align="center">
  <tr>
    <th style="text-align:center;">Low Resolution Input (64x64)</th>
    <th style="text-align:center;">ESRGAN Output (256x256)</th>
  </tr>
  <tr>
    <td align="center"><img src="Screenshot 2026-07-02 132321.png" width="256" alt="LR Leaf"></td>
    <td align="center"><img src="Screenshot 2026-07-02 132329.jpg" width="256" alt="SR Leaf"></td>
  </tr>
  <tr>
    <td align="center"><img src="Screenshot 2026-07-02 132416.png" width="256" alt="LR Bird"></td>
    <td align="center"><img src="Screenshot 2026-07-02 132424.jpg" width="256" alt="SR Bird"></td>
  </tr>
  <tr>
    <td align="center"><img src="Screenshot 2026-07-02 132453.png" width="256" alt="LR Flower"></td>
    <td align="center"><img src="Screenshot 2026-07-02 132502.jpg" width="256" alt="SR Flower"></td>
  </tr>
</table>

<hr>

<h2>🧠 Model Architecture</h2>

<p>This implementation faithfully reproduces the core improvements introduced by the official ESRGAN paper over standard SRGANs.</p>

<h3>1. The Generator (RRDB Net)</h3>
<p>The Generator is built to hallucinate missing high-frequency details.</p>
<ul>
  <li><strong>No Batch Normalization:</strong> BN layers are completely removed to prevent unpleasant artifact generation and improve computation speed.</li>
  <li><strong>Residual-in-Residual Dense Blocks (RRDB):</strong> Instead of standard residual blocks, the network uses a deeper, more complex RRDB structure. Each block contains dense connections where the output of every layer is concatenated to all subsequent layers.</li>
  <li><strong>PixelShuffle Upsampling:</strong> Uses sub-pixel convolution (<code>nn.PixelShuffle(2)</code>) to physically expand the spatial dimensions of the feature maps by a factor of 4x.</li>
</ul>

<h3>2. The Discriminator (VGG-Style)</h3>
<p>A VGG-style deep convolutional network with LeakyReLU activations. It analyzes patches of the image to determine if the generated textures look like real-world photographic textures or fake AI-generated noise.</p>

<h3>3. The Loss Function</h3>
<p>The model is trained using a highly balanced three-part loss function to ensure both color accuracy and photorealistic textures. The total Generator loss is calculated as:</p>

<div align="center">
  <code>L<sub>G</sub> = &lambda;<sub>1</sub>L<sub>pixel</sub> + &lambda;<sub>2</sub>L<sub>perceptual</sub> + &lambda;<sub>3</sub>L<sub>GAN</sub></code>
</div>
<br>

<ul>
  <li><strong>Pixel Loss (L<sub>pixel</sub>):</strong> Mean Absolute Error (L1) to ensure overall color and structural alignment.</li>
  <li><strong>Perceptual Loss (L<sub>perceptual</sub>):</strong> Extracts feature maps from a pre-trained <strong>VGG-19</strong> network. Instead of comparing pixels, it compares the high-level "style" and "texture" representations between the generated and real images.</li>
  <li><strong>Adversarial Loss (L<sub>GAN</sub>):</strong> Standard BCEWithLogitsLoss to fool the Discriminator.</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<p>This project is built with a strictly modular architecture to keep training loops clean and debuggable.</p>

<ul>
  <li><code>networks.py</code> - Contains the PyTorch classes for RRDB, DenseBlock, Generator, and Discriminator.</li>
  <li><code>losses.py</code> - Contains the custom GLoss and DLoss functions, balancing L1, VGG-19, and BCE losses.</li>
  <li><code>dataset.py</code> - Defines the SRDataset PyTorch dataloader, handling Bicubic downsampling and tensor transformations.</li>
  <li><code>train.ipynb</code> - The primary training pipeline (includes L1 pre-training and GAN training phases).</li>
  <li><code>test.ipynb</code> - Inference script for upscaling custom images.</li>
  <li><code>downsample.py</code> - Utility script for preprocessing and intentionally degrading real-world photos for testing.</li>
</ul>

<hr>

<h2>🚀 Installation & Usage</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.9+</li>
  <li>PyTorch & Torchvision (CUDA highly recommended)</li>
  <li>Pillow, Matplotlib, tqdm</li>
</ul>

<pre><code>git clone https://github.com/yourusername/ESRGAN-PyTorch.git
cd ESRGAN-PyTorch
pip install -r requirements.txt
</code></pre>

<h3>Inference (Testing a custom image)</h3>
<p>Open <code>test.ipynb</code> to upscale a single image. The pipeline will automatically resize your input to a clean 256x256 image to remove JPEG compression artifacts, and then push it through the Generator to output a stunning 1024x1024 image.</p>

<hr>

<h2>⚙️ Hardware Optimization Notes</h2>

<p>Training an ESRGAN is highly computationally expensive due to the depth of the VGG-19 feature extractor and the RRDB blocks.</p> 

<p>This repository is specifically optimized to run efficiently on consumer-grade hardware like the NVIDIA RTX 3050 (6GB). To avoid CUDA Out-of-Memory (OOM) errors, the dataloader is strictly balanced in <code>train.ipynb</code>:</p>

<ul>
  <li><strong>Recommended setting:</strong> <code>patch_size=128</code> with <code>batch_size=4</code></li>
  <li><strong>For larger patches:</strong> <code>patch_size=256</code> with <code>batch_size=1</code> or <code>2</code></li>
</ul>

<hr>

<h2>📝 License & Author</h2>
<p>Created by <strong>Jeet Mondal</strong>.</p>
<p>This project is open-source and available under the MIT License.</p>
