import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import argparse

def preprocess_data(input_path, output_path):
    print("Memulai proses otomatisasi...")
    if not os.path.exists(input_path):
        print(f"Error: File input tidak ditemukan di '{input_path}'")
        return

    try:
        df = pd.read_csv(input_path)
        print(f"Data berhasil dimuat dari '{input_path}'. Shape: {df.shape}")
    except Exception as e:
        print(f"Gagal memuat data: {e}")
        return

    # 0. Membuat kolom first_winner: 1 jika Winner == First_pokemon_id, 0 jika tidak
    df['first_winner'] = (df['Winner'] == df['First_pokemon_id']).astype(int)
    
    # 1. Drop kolom yang tidak diperlukan (sesuaikan dengan template)
    drop_cols = [
        'Name_first', 'Name_second', 'Winner', 'First_pokemon_id', 'Second_pokemon_id'
    ]
    df_cleaned = df.drop(columns=drop_cols, errors='ignore').drop_duplicates().copy()

    # 2. Label Encoding untuk kolom kategorikal
    cat_cols = [
        'Type 1_first', 'Type 2_first', 'Legendary_first',
        'Type 1_second', 'Type 2_second', 'Legendary_second'
    ]
    for col in cat_cols:
        if col in df_cleaned.columns:
            df_cleaned.loc[:, col] = LabelEncoder().fit_transform(df_cleaned[col].astype(str))

    # 3. Normalisasi (MinMaxScaler) untuk seluruh kolom fitur
    feature_cols = [col for col in df_cleaned.columns]
    scaler = MinMaxScaler()
    df_cleaned.loc[:, feature_cols] = scaler.fit_transform(df_cleaned[feature_cols])

    # 4. Simpan hasil ke CSV
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_cleaned.to_csv(output_path, index=False)
    print(f"Data yang sudah diproses berhasil disimpan di '{output_path}'. Shape: {df_cleaned.shape}")

    print("Proses otomatisasi selesai.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script untuk preprocessing data Pokemon.')
    parser.add_argument('--input', type=str, required=True, help='Path ke file CSV data mentah.')
    parser.add_argument('--output', type=str, required=True, help='Path untuk menyimpan file CSV yang sudah diproses.')
    args = parser.parse_args()

    preprocess_data(args.input, args.output)
