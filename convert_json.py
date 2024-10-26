import json
import sys

def update_path_in_json(input_file, output_file, new_path):
    # Read the input JSON file
    try:
        with open(input_file, 'r') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Input file {input_file} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the input file {input_file}.")
        sys.exit(1)

    # Update the path
    try:
        data['datasets'][0]['path'] = new_path
    except KeyError as e:
        print(f"Key error: {e}")
        sys.exit(1)
    except IndexError as e:
        print(f"Index error: {e}")
        sys.exit(1)

    # Write the updated JSON to the output file
    try:
        with open(output_file, 'w') as outfile:
            json.dump(data, outfile, indent=4)
    except IOError as e:
        print(f"Error writing to the output file {output_file}: {e}")
        sys.exit(1)

    print(f"Path updated successfully in {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert JSON for fine-tuning.')
    parser.add_argument('--input_json', type=str, help='Name of the dataset', required=True)
    parser.add_argument('--output_json', type=str, help='Path to the input CSV', required=True)
    parser.add_argument('--new_path', type=str, help='Path to output JSON', required=True)
    args = parser.parse_args()
    update_path_in_json(args.input_json, args.output_json, args.new_path)