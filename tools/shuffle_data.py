import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

def parse_timestamp_from_filename(filename):
    """Extract datetime from filename format: YYYYMMDDHH-typhoon-id.png"""
    try:
        timestamp_str = filename.split('-')[0]  # Get YYYYMMDDHH part
        return datetime.strptime(timestamp_str, '%Y%m%d%H')
    except (ValueError, IndexError):
        return None

def find_continuous_sequences(file_timestamps):
    """Split files into continuous sequences where time difference is exactly 1 hour"""
    sequences = []
    current_sequence = []
    
    # Sort files by timestamp
    sorted_files = sorted(file_timestamps.items(), key=lambda x: x[1])
    
    for i, (file_path, timestamp) in enumerate(sorted_files):
        if not current_sequence:
            current_sequence.append((file_path, timestamp))
            continue
            
        prev_timestamp = current_sequence[-1][1]
        time_diff = timestamp - prev_timestamp
        
        # Check if continuous (exactly 1 hour difference)
        if time_diff == timedelta(hours=1):
            current_sequence.append((file_path, timestamp))
        else:
            # End current sequence and start new one
            if len(current_sequence) >= 3:
                sequences.append([item[0] for item in current_sequence])
            current_sequence = [(file_path, timestamp)]
    
    # Don't forget the last sequence
    if len(current_sequence) >= 3:
        sequences.append([item[0] for item in current_sequence])
    
    return sequences

def generate_triplets(sequence):
    """Generate [t-1, t, t+1] triplets from continuous sequence"""
    triplets = []
    for i in range(1, len(sequence) - 1):
        triplet = [sequence[i-1], sequence[i], sequence[i+1]]
        triplets.append(triplet)
    return triplets

def main(root_dir, output_json='triplets.json'):
    all_triplets = []
    
    # Walk through all typhoon directories
    for typhoon_dir in os.listdir(root_dir):
        typhoon_path = os.path.join(root_dir, typhoon_dir)
        
        if not os.path.isdir(typhoon_path):
            continue
            
        print(f"Processing typhoon: {typhoon_dir}")
        file_timestamps = {}
        
        # Collect all valid files with their timestamps
        for filename in os.listdir(typhoon_path):
            if filename.endswith('.png'):
                file_path = os.path.join(typhoon_path, filename)
                timestamp = parse_timestamp_from_filename(filename)
                
                if timestamp:
                    file_timestamps[file_path] = timestamp
        
        if not file_timestamps:
            continue
            
        # Find continuous sequences
        sequences = find_continuous_sequences(file_timestamps)
        
        # Generate triplets for each sequence
        for sequence in sequences:
            triplets = generate_triplets(sequence)
            all_triplets.extend(triplets)
        
        print(f"  Found {len(sequences)} continuous sequences, generated {len(triplets)} triplets")
    
    # Save triplets to JSON
    with open(output_json, 'w') as f:
        json.dump(all_triplets, f, indent=2)
    
    print(f"\nTotal triplets generated: {len(all_triplets)}")
    print(f"Triplets saved to: {output_json}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate temporal triplets for TempoGAN training')
    parser.add_argument('--root_dir', type=str, required=True, 
                       help='Root directory containing typhoon folders')
    parser.add_argument('--output', type=str, default='triplets.json',
                       help='Output JSON file path (default: triplets.json)')
    
    args = parser.parse_args()
    main(args.root_dir, args.output)