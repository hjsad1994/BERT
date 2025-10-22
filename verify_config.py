import yaml

print("Verifying config...")
config = yaml.safe_load(open('multi_label/config_multi.yaml'))
print("Config loaded successfully!")
print(f"\nCurrent values:")
print(f"  Focal weight: {config['multi_label']['focal_weight']}")
print(f"  Contrastive weight: {config['multi_label']['contrastive_weight']}")
print(f"  Temperature: {config['multi_label']['contrastive_temperature']}")
print(f"  Epochs: {config['training']['num_train_epochs']}")
print(f"  Output dir: {config['paths']['output_dir']}")
print("\nAll configs accessible!")
