#!/usr/bin/env python3
"""
API test - JSON çıktısını gösterir
"""
import requests
import json

response = requests.get('http://localhost:5000/api/traffic')
data = response.json()

print(f"Timestamp: {data['timestamp']}")
print(f"Feature sayısı: {len(data['geojson']['features'])}")
print()

if len(data['geojson']['features']) > 0:
    print("İlk feature:")
    print(json.dumps(data['geojson']['features'][0], indent=2))
else:
    print("⚠️  Hiç feature yok!")
    
print()
print("Hata var mı:")
print(data.get('error', 'Yok'))
