#!/bin/bash
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
case $1 in
    -h|--help)
        echo "Usage: download_checkpoints.sh [-h|--help] [-t|--toy]"
        echo "Downloads checkpoints from archive.org and saves them to the checkpoints directory."
        echo "  -h, --help              Print this help message and exit"
        echo "  -t, --toy               Download checkpoints"
        exit 0
        ;;
    -t|--toy)
        TOY="TRUE"
        shift
        ;;
    *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
esac
done

if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

if [ -n "$TOY" ]; then
    echo "Downloading toy checkpoints..."
    wget -O checkpoints/ViT_toy\ 3x3.zip https://archive.org/download/vision-diffmask-ckpts/ViT_toy%203x3.zip
    unzip checkpoints/ViT_toy\ 3x3.zip -d checkpoints/
    rm checkpoints/ViT_toy\ 3x3.zip
fi

wget -O checkpoints/diffmask.ckpt https://archive.org/download/vision-diffmask-ckpts/diffmask.ckpt



