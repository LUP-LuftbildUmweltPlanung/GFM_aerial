import torch

# Lade das Swin-Transformer Modell (Standard-Teacher)
#swin_teacher = SwinTransformer(r"GFM-main/output/simmim_finetune/swin_base_patch4_window7_224_22k.pth")  # Initialisiere mit den originalen Parametern
swin_teacher = torch.load("GFM-main/output/simmim_finetune/swin_base_patch4_window7_224_22k.pth", map_location="cpu")  # Initialisiere mit den originalen Parametern
#swin_state_dict = swin_teacher.state_dict().keys()
swin_state_dict = swin_teacher.keys()
#swin_state_dict = swin_teacher['model']

print(swin_state_dict)
# Lade das vortrainierte GFM-Modell
gfm_teacher = torch.load("GFM-main/output/simmim_finetune/gfm.pth", map_location="cpu")
gfm_state_dict = gfm_teacher.keys()  # Die Keys aus dem `state_dict`
#gfm_state_dict = gfm_teacher['model'] # Die Keys aus dem `state_dict`
gfm_config = gfm_teacher['config']
#print(gfm_state_dict)
#print("\n")
print(gfm_config)
print("\n")
print(gfm_teacher['optimizer'])
exit()
# Vergleiche die Keys
#swin_only = swin_state_dict - gfm_state_dict
#gfm_only = gfm_state_dict - swin_state_dict
swin_keys = set(swin_state_dict.keys())
gfm_keys = set(gfm_state_dict.keys())

# Vergleiche die Layer-Namen
only_in_swin = swin_keys - gfm_keys
only_in_gfm = gfm_keys - swin_keys
common_keys = swin_keys & gfm_keys
#print(f"🔍 Keys, die nur im Swin-Modell sind: {swin_only}")
#print(f"🔍 Keys, die nur im GFM-Modell sind: {gfm_only}")
print(f"🔍 {len(common_keys)} gemeinsame Layer")
print(f"🚨 {len(only_in_swin)} Layer nur in Swin: {only_in_swin}")
print(f"🚨 {len(only_in_gfm)} Layer nur in GFM: {only_in_gfm}")