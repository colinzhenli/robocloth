PROJECT="${1:?usage: $0 <project_dir> <gpu_id>}"
gpu_id="${2:?usage: $0 <project_dir> <gpu_id>}"

# === LOCAL STORAGE SETUP (avoid slow NFS I/O) ===
TMP_BASE="${COLMAP_TMP:-/tmp/robocloth_colmap}"
PROJECT_NAME=$(basename "${PROJECT}")
TMP_PROJECT="${TMP_BASE}/${PROJECT_NAME}_$$"  # $$ = PID for uniqueness

echo "== Setting up local storage =="
echo "Original project: ${PROJECT}"
echo "Temp project: ${TMP_PROJECT}"

# Create tmp project directory
mkdir -p "${TMP_PROJECT}"

# Copy images to local storage
echo "== Copying images to local storage =="
cp -r "${PROJECT}/ldr" "${TMP_PROJECT}/ldr"
echo "Images copied to ${TMP_PROJECT}/ldr"

# Define paths using local storage
IMG_DIR="${TMP_PROJECT}/ldr"
DB="${TMP_PROJECT}/database.db"
OUT_SPARSE="${TMP_PROJECT}/sparse"

echo "== Feature extraction =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMG_DIR" \
    --ImageReader.single_camera=true \

echo "== Exhaustive matching =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap exhaustive_matcher --database_path "$DB"

echo "== Mapping =="
mkdir -p ${TMP_PROJECT}/sparse
CUDA_VISIBLE_DEVICES=$gpu_id colmap mapper \
    --database_path=${TMP_PROJECT}/database.db \
    --image_path=${IMG_DIR} \
    --output_path=${TMP_PROJECT}/sparse

echo "== Using submodel 0 (largest) as final reconstruction =="
# model_merger silently fails when submodels don't share images, producing a
# corrupt merged model. Keep only sparse/0 (the largest submodel).
cp ${TMP_PROJECT}/sparse/0/*.bin ${TMP_PROJECT}/sparse/

echo "== Convert sparse model to text =="
colmap model_converter \
    --input_path   ${TMP_PROJECT}/sparse \
    --output_path  ${TMP_PROJECT}/sparse \
    --output_type  TXT

echo "== Convert sparse model to ply =="
CUDA_VISIBLE_DEVICES=$gpu_id colmap model_converter \
    --input_path "${TMP_PROJECT}/sparse" \
    --output_path "${TMP_PROJECT}/sparse/points3D.ply" \
    --output_type PLY

# === MOVE RESULTS BACK TO NFS ===
# Back up the existing (failed) sparse dir before replacing it
echo "== Moving sparse output back to original project =="
if [ -d "${PROJECT}/sparse" ]; then
    BACKUP="${PROJECT}/sparse_seq_failed"
    rm -rf "${BACKUP}" 2>/dev/null
    mv "${PROJECT}/sparse" "${BACKUP}"
    echo "Backed up old sparse to ${BACKUP}"
fi
mv "${TMP_PROJECT}/sparse" "${PROJECT}/sparse"
echo "Sparse output moved to ${PROJECT}/sparse"

# === CLEANUP ===
echo "== Cleaning up temporary files =="
rm -rf "${TMP_PROJECT}"
echo "Temporary directory removed: ${TMP_PROJECT}"

echo "== Finished COLMAP reconstruction =="
