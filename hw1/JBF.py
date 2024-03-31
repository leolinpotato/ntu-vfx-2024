import numpy as np
import cv2

def bilateral_filter(E, low_w=0.4, high_w=1.0, sigma_s=1, sigma_r=0.05):
    B, G, R = 0.06, 0.67, 0.27
    intensity = (E[:,:,0]*B + E[:,:,1]*G + E[:,:,2]*R).astype(np.float32)
    color = E/intensity[..., np.newaxis]
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    large_scale = JBF.joint_bilateral_filter(intensity, intensity).astype(np.float32)
    detail_map = intensity/large_scale
    epsilon = 1e-10
    large_scale_log = np.log2(large_scale + epsilon)
    detail_map_log = np.log2(detail_map + epsilon)
    combined_map = np.power(2, low_w*large_scale_log+high_w*detail_map)
    output = color * combined_map[..., np.newaxis]
    output = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype('uint8')
    return output

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        ### TODO ###

        # normalize the guidance image
        padded_guidance = padded_guidance.astype('float64') / 255
        padded_img = padded_img.astype('float64')

        # calculate Gs(constant)
        Gs = np.array([[np.exp(-((i - self.pad_w)**2 + (j - self.pad_w)**2) / (2 * self.sigma_s**2)) for i in range(self.wndw_size)] for j in range(self.wndw_size)])
    
        # calculate Gr and I'p
        output = np.zeros(img.shape)
        for i in range(self.pad_w, padded_guidance.shape[0] - self.pad_w):
            for j in range(self.pad_w, padded_guidance.shape[1] - self.pad_w):
                Tp = padded_guidance[i, j]
                Tq = padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]
                Ip = padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]

                power = -((Tp - Tq)**2 / (2 * self.sigma_r**2))
                if len(power.shape) == 3:
                    power = np.sum(power, axis=2)
                Gr = np.exp(power)
                G = np.multiply(Gs, Gr)
                W = G.sum()

                # G_expanded = np.expand_dims(G, axis=-1)
                # output[i - self.pad_w][j - self.pad_w] = np.sum(G_expanded * Ip, axis=(0, 1)) / W
                if len(img.shape) == 2:
                    output[i-self.pad_w, j-self.pad_w] = np.multiply(G, Ip[:,:]).sum() / W    
                else:
                    output[i-self.pad_w, j-self.pad_w] = [np.multiply(G, Ip[:,:,k]).sum() / W for k in range(img.shape[2])]

        
        return output
