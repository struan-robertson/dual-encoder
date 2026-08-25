for file in run_*.toml; do
    echo "run $file"
    python src/training.py "$file"
done
