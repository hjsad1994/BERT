# -*- coding: utf-8 -*-
"""
Script test logic oversampling
"""
import sys
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Gia lap data de test logic
positive = 6000
negative = 3500
neutral = 500
total = positive + negative + neutral

print('=== DU LIEU GOC ===')
print(f'Positive: {positive} ({positive/total*100:.1f}%)')
print(f'Negative: {negative} ({negative/total*100:.1f}%)')
print(f'Neutral:  {neutral} ({neutral/total*100:.1f}%)')
print(f'Imbalance ratio: {positive/neutral:.2f}x')

# Logic trong train.py
majority_count = max(positive, negative, neutral)
target_neutral_count = int(majority_count * 0.4)

print(f'\n=== OVERSAMPLING STRATEGY ===')
print(f'Majority count: {majority_count}')
print(f'Target neutral: {target_neutral_count} (40% of majority)')

# Sau oversampling
positive_after = positive
negative_after = negative
neutral_after = max(target_neutral_count, neutral)
total_after = positive_after + negative_after + neutral_after

print(f'\n=== SAU OVERSAMPLING ===')
print(f'Positive: {positive_after} ({positive_after/total_after*100:.1f}%)')
print(f'Negative: {negative_after} ({negative_after/total_after*100:.1f}%)')
print(f'Neutral:  {neutral_after} ({neutral_after/total_after*100:.1f}%)')
print(f'Imbalance ratio: {positive_after/neutral_after:.2f}x')

# Focal Loss alpha tinh tren data SAU oversampling (SAI!)
print(f'\n=== FOCAL LOSS ALPHA (HIEN TAI - SAI!) ===')
alpha_pos = total_after / (3 * positive_after)
alpha_neg = total_after / (3 * negative_after)
alpha_neu = total_after / (3 * neutral_after)
print(f'Alpha positive: {alpha_pos:.4f}')
print(f'Alpha negative: {alpha_neg:.4f}')
print(f'Alpha neutral:  {alpha_neu:.4f}')
print('=> Alpha gan bang nhau, Focal Loss mat tac dung!')

# Focal Loss alpha nen tinh tren data GOC (DUNG!)
print(f'\n=== FOCAL LOSS ALPHA (NEN LA - DUNG!) ===')
alpha_pos_correct = total / (3 * positive)
alpha_neg_correct = total / (3 * negative)
alpha_neu_correct = total / (3 * neutral)
print(f'Alpha positive: {alpha_pos_correct:.4f}')
print(f'Alpha negative: {alpha_neg_correct:.4f}')
print(f'Alpha neutral:  {alpha_neu_correct:.4f}')
print('=> Alpha phan anh dung imbalance goc!')

print(f'\n=== KET LUAN ===')
print('LOI NGHIEM TRONG: Focal Loss alpha dang duoc tinh tren du lieu SAU oversampling!')
print('=> Phai tinh alpha TRUOC khi oversample de giu nguyen imbalance weight!')
