path=torch_random_fields/models

echo 'Running autoflake ...'
find $path -type f -name "*.py" | xargs autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports 

echo 'Running isort ...'
find $path -type f -name "*.py" | xargs isort

# echo 'Running yapf ...'
# find $path -type f -name "*.py" | xargs yapf -i