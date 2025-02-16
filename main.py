from src.interface import Interface


if __name__ == "__main__":
    interface = Interface()
    interface.run_interface()

# TODO 
# Concatenate accepted recommendations with validated transactions
# Reduce image size to speed up data reading
# Read transactions + proofs files in parallel
# Download validated transactions