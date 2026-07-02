import torch

def GLoss(sr_img,hr_img,sr_feature_map,feature_extractor,L1,BCEloss):
    l1_loss = L1(sr_img,hr_img)      # Pixel loss

    adversarial_loss = BCEloss(sr_feature_map, torch.ones_like(sr_feature_map))      # GAN loss

    sr_feature = feature_extractor(sr_img)
    hr_feature = feature_extractor(hr_img).detach()

    perceptual_loss = L1(sr_feature, hr_feature)     # Perceptual loss

    return 0.01*l1_loss + 0.005*adversarial_loss + 0.1*perceptual_loss

def DLoss(D,sr_img,hr_img,BCELoss):
    real_feature = D(hr_img.detach())
    fake_feature = D(sr_img.detach())

    real_loss = BCELoss(real_feature, torch.ones_like(real_feature))
    fake_loss = BCELoss(fake_feature, torch.zeros_like(fake_feature))

    return real_loss + fake_loss