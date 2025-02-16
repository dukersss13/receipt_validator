from src.interface import Interface


if __name__ == "__main__":
    interface = Interface()
    interface.run_interface()

# TODO
# Reduce image size to speed up data reading
# Read transactions + proofs files in parallel
# Download validated transactions
# Enable PDF read