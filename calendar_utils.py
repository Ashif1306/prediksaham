"""
calendar_utils.py
=================
Utility functions untuk penanganan kalender perdagangan Bursa Efek Indonesia (BEI).

Modul ini menyediakan fungsi-fungsi untuk:
- Mengelola daftar hari libur BEI 2026
- Menentukan apakah suatu tanggal adalah hari perdagangan
- Mencari hari perdagangan berikutnya (skip weekend & holiday)

Author: [Your Name]
Date: 2026-02-16
Version: 1.0
"""

import pandas as pd
from datetime import datetime, timedelta


# =============================================================================
# KONFIGURASI HARI LIBUR BEI 2026
# =============================================================================

# Daftar resmi hari libur Bursa Efek Indonesia tahun 2026
# Total: 22 hari (libur nasional + cuti bersama)
BEI_HOLIDAYS_2026 = pd.to_datetime([
    '2026-01-01',  # Tahun Baru Masehi
    '2026-01-16',  # Cuti Bersama Tahun Baru Imlek
    '2026-02-16',  # Tahun Baru Imlek 2577 Kongzili
    '2026-02-17',  # Tahun Baru Imlek 2577 Kongzili
    '2026-03-18',  # Hari Suci Nyepi Tahun Baru Saka 1948
    '2026-03-19',  # Cuti Bersama Hari Suci Nyepi
    '2026-03-20',  # Cuti Bersama Hari Suci Nyepi
    '2026-03-23',  # Isra Mikraj Nabi Muhammad SAW
    '2026-03-24',  # Cuti Bersama Isra Mikraj
    '2026-04-03',  # Wafat Isa Al-Masih
    '2026-05-01',  # Hari Buruh Internasional
    '2026-05-14',  # Kenaikan Isa Al-Masih
    '2026-05-15',  # Cuti Bersama Hari Raya Idul Fitri
    '2026-05-27',  # Hari Raya Idul Fitri 1447 Hijriah
    '2026-05-28',  # Hari Raya Idul Fitri 1447 Hijriah
    '2026-06-01',  # Hari Lahir Pancasila
    '2026-06-16',  # Hari Raya Waisak 2570
    '2026-08-17',  # Hari Kemerdekaan RI
    '2026-08-25',  # Cuti Bersama Hari Raya Idul Adha
    '2026-12-24',  # Cuti Bersama Hari Raya Natal
    '2026-12-25',  # Hari Raya Natal
    '2026-12-31',  # Cuti Bersama Tahun Baru 2027
])


# =============================================================================
# FUNGSI UTILITAS KALENDER
# =============================================================================

def is_trading_day(date):
    """
    Memeriksa apakah suatu tanggal adalah hari perdagangan.
    
    Hari perdagangan adalah hari Senin-Jumat yang bukan termasuk
    dalam daftar hari libur BEI.
    
    Args:
        date (datetime or str): Tanggal yang akan dicek.
                                Bisa berupa datetime object atau string 'YYYY-MM-DD'.
    
    Returns:
        bool: True jika tanggal adalah hari perdagangan, False jika bukan.
    
    Example:
        >>> is_trading_day('2026-02-14')  # Jumat biasa
        True
        >>> is_trading_day('2026-02-15')  # Sabtu (weekend)
        False
        >>> is_trading_day('2026-02-16')  # Senin (libur Imlek)
        False
    """
    # Konversi ke datetime jika input berupa string
    if isinstance(date, str):
        date = pd.to_datetime(date)
    
    # Cek apakah hari weekend (Sabtu=5, Minggu=6)
    if date.weekday() >= 5:
        return False
    
    # Normalisasi tanggal untuk perbandingan (set waktu ke 00:00:00)
    date_normalized = pd.Timestamp(date).normalize()
    
    # Cek apakah tanggal termasuk dalam daftar hari libur BEI
    if date_normalized in BEI_HOLIDAYS_2026:
        return False
    
    # Jika bukan weekend dan bukan libur, maka adalah hari perdagangan
    return True


def get_next_trading_day(last_date, max_days_ahead=30):
    """
    Menentukan hari perdagangan berikutnya setelah tanggal tertentu.
    
    Fungsi ini akan mencari hari perdagangan pertama setelah last_date
    dengan melewati weekend (Sabtu-Minggu) dan hari libur BEI.
    
    Args:
        last_date (datetime or str): Tanggal terakhir data historis.
                                     Bisa berupa datetime object atau string 'YYYY-MM-DD'.
        max_days_ahead (int, optional): Maksimal hari ke depan untuk dicek.
                                        Default: 30 hari.
    
    Returns:
        datetime: Tanggal hari perdagangan berikutnya.
    
    Raises:
        ValueError: Jika tidak ditemukan hari perdagangan dalam max_days_ahead hari.
    
    Example:
        >>> get_next_trading_day('2026-02-14')
        Timestamp('2026-02-18 00:00:00')  # Rabu (skip weekend + Imlek 16-17)
        
        >>> get_next_trading_day('2026-03-17')
        Timestamp('2026-03-25 00:00:00')  # Rabu (skip Nyepi 18-20 + Isra Mikraj 23-24)
    """
    # Konversi ke datetime jika input berupa string
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    # Mulai pencarian dari hari berikutnya
    next_date = last_date + timedelta(days=1)
    
    # Loop mencari hari perdagangan berikutnya
    for _ in range(max_days_ahead):
        # Cek apakah next_date adalah hari perdagangan
        if is_trading_day(next_date):
            return next_date
        
        # Jika bukan, lanjut ke hari berikutnya
        next_date += timedelta(days=1)
    
    # Jika tidak ditemukan dalam max_days_ahead hari
    raise ValueError(
        f"Tidak ditemukan hari perdagangan dalam {max_days_ahead} hari ke depan "
        f"dari tanggal {last_date.date()}"
    )


def get_trading_days_between(start_date, end_date):
    """
    Menghitung jumlah hari perdagangan antara dua tanggal.
    
    Fungsi utilitas tambahan untuk menghitung berapa hari perdagangan
    antara start_date dan end_date (inklusif).
    
    Args:
        start_date (datetime or str): Tanggal mulai.
        end_date (datetime or str): Tanggal akhir.
    
    Returns:
        int: Jumlah hari perdagangan antara kedua tanggal.
    
    Example:
        >>> get_trading_days_between('2026-02-14', '2026-02-20')
        3  # Jumat 14, Rabu 18, Kamis 19, Jumat 20 = 4 hari
             # (skip Sabtu 15, Minggu 16, Senin 17 libur, Selasa 18 libur)
    """
    # Konversi ke datetime
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Counter hari perdagangan
    trading_days_count = 0
    current_date = start_date
    
    # Loop dari start_date sampai end_date
    while current_date <= end_date:
        if is_trading_day(current_date):
            trading_days_count += 1
        current_date += timedelta(days=1)
    
    return trading_days_count


# =============================================================================
# TESTING & VALIDASI (OPSIONAL)
# =============================================================================

def validate_calendar():
    """
    Fungsi untuk memvalidasi konfigurasi kalender.
    Berguna untuk testing dan debugging.
    
    Returns:
        dict: Informasi validasi kalender.
    """
    print("="*70)
    print("VALIDASI KALENDER TRADING BEI 2026")
    print("="*70)
    
    print(f"\nTotal hari libur BEI 2026: {len(BEI_HOLIDAYS_2026)} hari")
    
    # Hitung hari perdagangan dalam setahun
    year_start = pd.to_datetime('2026-01-01')
    year_end = pd.to_datetime('2026-12-31')
    
    total_days = (year_end - year_start).days + 1
    trading_days = get_trading_days_between(year_start, year_end)
    weekends = sum(1 for d in pd.date_range(year_start, year_end) if d.weekday() >= 5)
    holidays = len(BEI_HOLIDAYS_2026)
    
    print(f"\nStatistik 2026:")
    print(f"  Total hari dalam setahun  : {total_days}")
    print(f"  Hari perdagangan (trading): {trading_days}")
    print(f"  Weekend (Sabtu-Minggu)    : {weekends}")
    print(f"  Hari libur BEI            : {holidays}")
    print(f"  Non-trading days total    : {weekends + holidays}")
    
    # Test beberapa kasus
    print(f"\nTest Case:")
    test_cases = [
        ('2026-02-14', 'Jumat sebelum Imlek'),
        ('2026-03-17', 'Selasa sebelum Nyepi'),
        ('2026-05-26', 'Selasa sebelum Lebaran'),
        ('2026-12-24', 'Kamis sebelum Natal'),
    ]
    
    for date_str, desc in test_cases:
        next_day = get_next_trading_day(date_str)
        days_skip = (next_day - pd.to_datetime(date_str)).days - 1
        print(f"  {date_str} ({desc})")
        print(f"    → Next trading: {next_day.date()} (skip {days_skip} hari)")
    
    print("="*70)
    
    return {
        'total_days': total_days,
        'trading_days': trading_days,
        'weekends': weekends,
        'holidays': holidays
    }


# =============================================================================
# MAIN (untuk testing manual)
# =============================================================================

if __name__ == "__main__":
    """
    Testing manual untuk memvalidasi fungsi-fungsi kalender.
    Jalankan: python calendar_utils.py
    """
    validate_calendar()
    
    print("\n" + "="*70)
    print("CONTOH PENGGUNAAN FUNGSI")
    print("="*70)
    
    # Contoh 1: Cek hari perdagangan
    print("\n1. Cek apakah tanggal adalah hari perdagangan:")
    test_dates = [
        '2026-02-13',  # Jumat biasa
        '2026-02-14',  # Sabtu (weekend)
        '2026-02-16',  # Senin (libur Imlek)
    ]
    
    for date in test_dates:
        result = is_trading_day(date)
        day_name = pd.to_datetime(date).strftime('%A')
        print(f"   {date} ({day_name:>9}): {'✓ Trading day' if result else '✗ Bukan trading day'}")
    
    # Contoh 2: Next trading day
    print("\n2. Mencari next trading day:")
    from_date = '2026-02-14'
    next_td = get_next_trading_day(from_date)
    print(f"   Dari tanggal    : {from_date}")
    print(f"   Next trading day: {next_td.date()} ({next_td.strftime('%A')})")
    
    # Contoh 3: Hitung trading days
    print("\n3. Hitung hari perdagangan antara dua tanggal:")
    start = '2026-02-01'
    end = '2026-02-28'
    count = get_trading_days_between(start, end)
    print(f"   Periode: {start} hingga {end}")
    print(f"   Trading days: {count} hari")
    
    print("="*70)