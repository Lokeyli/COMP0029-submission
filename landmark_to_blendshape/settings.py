BLENDSHAPE_FILE = "../data-generation-demo/blendshapes.csv"       # Modify if necessary
TRAIN_FILE = "../data-generation-demo/train.csv"                  # Modify if necessary
N_LANDMARKS = 478
N_BLENDSHAPE = 48
FIRST_BLENDSHAPE_IDX = 67
HEADERS = ["blendshape_i", "weight"]
HEADERS.extend([f"landmark_{i}" for i in range(N_LANDMARKS)])

if __name__ == "__main__":
    print(HEADERS)
    print(len(HEADERS))
