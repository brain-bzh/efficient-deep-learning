import argparse
import os

def main():    
    parser = argparse.ArgumentParser(description="Deep Learning Pipeline")
    parser.add_argument("mode", choices=["train", "evaluate","binary_train","pruning_train"], help="Mode to run the script")
    parser.add_argument("--data_path", type=str, default="/opt/img/effdl-cifar10/", help="Path to the dataset")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for regularization")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="models/cnn_model.pth", help="Path to save the model")

    # Pruning 
    parser.add_argument("--amount", type=float, default=0.2, help="amount for global unstructured pruning")

    # Evaluation arguments
    parser.add_argument("--model_path", type=str, default="models/test.pth", help="Path to the trained model")

    args = parser.parse_args()

    if args.mode == "train":
        os.system(f"python src/train.py --epochs {args.epochs} --weight_decay {args.weight_decay} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --data_path {args.data_path} --save_path {args.save_path}")
    elif args.mode == "evaluate":
        os.system(f"python src/evaluate.py --model_path {args.model_path} --batch_size {args.batch_size} --data_path {args.data_path}")

    elif args.mode == "binary_train":
        os.system(f"python src/binary_train.py --epochs {args.epochs} --weight_decay {args.weight_decay} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --data_path {args.data_path} --save_path {args.save_path}")

    elif args.mode == "pruning_train":
        os.system(f"python src/pruning_train.py --epochs {args.epochs} --weight_decay {args.weight_decay} --batch_size {args.batch_size} --learning_rate {args.learning_rate} --data_path {args.data_path} --amount {args.amount}")

if __name__ == "__main__":
    main()