import sys
import os
import great_expectations as gx

# Define the marker file path
SUCCESS_MARKER = "validation_success.txt"

def validate_data():
    context = gx.get_context()

    print("Running Checkpoint 'fraud_checkpoint'...")
    results = context.run_checkpoint(checkpoint_name="fraud_checkpoint")

    if not results["success"]:
        print("❌ Validation Failed!")
        sys.exit(1)
    
    print("✅ Validation Succeeded!")
    
    # --- NEW: Write the Success Marker ---
    with open(SUCCESS_MARKER, "w") as f:
        f.write("Validation passed successfully.")
    print(f"Marker file created at: {SUCCESS_MARKER}")

if __name__ == "__main__":
    validate_data()