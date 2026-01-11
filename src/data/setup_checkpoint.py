import great_expectations as gx

def setup_checkpoint():
    print("Setting up Great Expectations Checkpoint...")
    
    # 1. Load your existing Data Context
    context = gx.get_context()

    # 2. Define the Data Asset (Connect to your CSV)
    datasource_name = "pandas_datasource"
    asset_name = "creditcard_asset"
    
    # Add/Update the datasource
    datasource = context.sources.add_or_update_pandas(datasource_name)
    asset = datasource.add_csv_asset(
        name=asset_name, 
        filepath_or_buffer="data/raw/creditcard.csv" 
    )

    # 3. Create an Expectation Suite (The rules)
    suite_name = "fraud_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
    
    # Get a validator to define expectations
    batch_request = asset.build_batch_request()
    validator = context.get_validator(
        batch_request=batch_request, 
        expectation_suite_name=suite_name
    )

    # --- ADD BASIC RULES HERE ---
    # Example: 'Class' column must exist
    validator.expect_column_to_exist("Class")
    # Example: 'Amount' shouldn't be null
    validator.expect_column_values_to_not_be_null("Amount")
    
    # Save the suite
    validator.save_expectation_suite(discard_failed_expectations=False)

    # 4. Define and Save the Checkpoint
    checkpoint_name = "fraud_checkpoint"
    checkpoint = context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite_name,
            },
        ],
    )
    
    print(f"✅ Checkpoint '{checkpoint_name}' created successfully!")

if __name__ == "__main__":
    setup_checkpoint()