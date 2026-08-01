import torch
import clip
import torch.nn.functional as F
from configs.eval_config import TEMPERATURE, WINDOW_SIZES

class WinCLIP:
    def __init__(self, model_name="ViT-B/16", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.temperature = TEMPERATURE
        self.window_sizes = WINDOW_SIZES

        self.normal_states = ['flawless', 'perfect', 'unblemished', 'without defect', 'intact', 'normal']
        
        self.anomaly_states = [
            'damaged', 'broken', 'defective', 'flawed', 'blemished', 'scratched', 
            'with a defect', 'with a crack', 'with a stain', 'with a hole', 
            'discolored', 'dented', 'bent', 'torn', 'missing a piece', 
            'cut', 'chipped', 'dirty', 'rusty', 'deformed'
        ]
        
        self.bg_states = ['dark empty space', 'a conveyor belt', 'an empty background', 'the background']

        self.templates = [
            'a cropped photo of the {} {}',
            'a close-up photo of a {} {}',
            'a photo of a {} {} for anomaly detection'
        ]

        self.degradations = {
            'clean': [''],
            'blur': ['blurry', 'out-of-focus'],
            'dark': ['dark', 'poorly-lit', 'underexposed'],
            'bright': ['bright', 'overexposed', 'glaring'],
            'grain': ['grainy', 'noisy', 'static']
        }

    def encode_text_cpe(self, class_name):
        """Builds a [5, 3, 512] anchor matrix (5 Degradations x 3 Classes)"""
        self.anchors = []

        for deg_key, deg_words in self.degradations.items():
            norm_prompts, anom_prompts, bg_prompts = [], [], []

            for deg in deg_words:
                prefix = f"{deg} " if deg else ""

                for template in self.templates:
                    for state in self.normal_states:
                        norm_prompts.append(template.format(f"{prefix}{state}", class_name))
                    for state in self.anomaly_states:
                        anom_prompts.append(template.format(f"{prefix}{state}", class_name))

                for bg in self.bg_states:
                    bg_prompts.append(f"a {prefix}photo of {bg}")

            with torch.no_grad():
                n_tok = clip.tokenize(norm_prompts).to(self.device)
                n_emb = self.model.encode_text(n_tok)
                n_emb /= n_emb.norm(dim=-1, keepdim=True)
                n_emb = n_emb.mean(dim=0, keepdim=True)
                n_emb /= n_emb.norm(dim=-1, keepdim=True)

                a_tok = clip.tokenize(anom_prompts[:77]).to(self.device)
                a_emb = self.model.encode_text(a_tok)
                a_emb /= a_emb.norm(dim=-1, keepdim=True)
                a_emb = a_emb.mean(dim=0, keepdim=True)
                a_emb /= a_emb.norm(dim=-1, keepdim=True)

                b_tok = clip.tokenize(bg_prompts).to(self.device)
                b_emb = self.model.encode_text(b_tok)
                b_emb /= b_emb.norm(dim=-1, keepdim=True)
                b_emb = b_emb.mean(dim=0, keepdim=True)
                b_emb /= b_emb.norm(dim=-1, keepdim=True)

                cond_matrix = torch.cat([n_emb, a_emb, b_emb], dim=0)
                self.anchors.append(cond_matrix)

        self.anchors = torch.stack(self.anchors).to(self.device)

    def extract_window_features(self, image, weights):
        B, C, H, W = image.shape

        with torch.no_grad():
            w = weights.view(5, 1, 1)
            T_adapted = (w * self.anchors).sum(dim=0)
            T_adapted /= T_adapted.norm(dim=-1, keepdim=True) 
            T_adapted = T_adapted.to(dtype=self.model.dtype)

            global_embed = self.model.encode_image(image.to(dtype=self.model.dtype))
            global_embed /= global_embed.norm(dim=-1, keepdim=True)
            
            global_logits = self.temperature * global_embed @ T_adapted.T
            global_prob_anomaly = global_logits[:, 0:2].softmax(dim=-1)[:, 1].item()
            global_prob_bg = global_logits.softmax(dim=-1)[:, 2].item()
            
            actual_global_score = global_prob_anomaly * (1.0 - global_prob_bg)

            max_global_score = 0.0

            for grid_size in self.window_sizes:
                stride = H // grid_size
                win_size = stride
                step = stride // 2

                crops = F.unfold(image, kernel_size=win_size, stride=step)
                crops = crops.transpose(1, 2).reshape(-1, C, win_size, win_size)
                crops_resized = F.interpolate(crops, size=(224, 224), mode='bicubic')

                win_embeds = []
                batch_size = 64 
                
                for idx in range(0, len(crops_resized), batch_size):
                    batch = crops_resized[idx : idx + batch_size]
                    embeds = self.model.encode_image(batch.to(dtype=self.model.dtype)) 
                    embeds /= embeds.norm(dim=-1, keepdim=True)
                    win_embeds.append(embeds)
                    
                win_embeds = torch.cat(win_embeds, dim=0)

                logits = self.temperature * win_embeds @ T_adapted.T
                prob_anomaly = logits[:, 0:2].softmax(dim=-1)[:, 1]
                prob_background = logits.softmax(dim=-1)[:, 2]

                semantic_weight = 1.0 - prob_background
                window_scores = prob_anomaly * semantic_weight

                crop_idx = 0
                for i in range(0, H - win_size + 1, step):
                    for j in range(0, W - win_size + 1, step):
                        s_weight = semantic_weight[crop_idx].item()
                        w_score = window_scores[crop_idx].item()

                        if s_weight > 0.01:
                            if w_score > max_global_score:
                                max_global_score = w_score

                        crop_idx += 1

            final_image_score = (actual_global_score + max_global_score) / 2.0

        return final_image_score