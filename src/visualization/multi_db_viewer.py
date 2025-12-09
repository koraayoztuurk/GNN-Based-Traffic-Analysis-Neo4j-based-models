#!/usr/bin/env python3
"""
multi_db_viewer.py
---------------------
Tüm veritabanlarının görselleştirmesini aynı anda başlat
Neo4j (port 5000), ArangoDB (port 5001), TigerGraph (port 5002)
"""
import sys
import time
import subprocess
from pathlib import Path

def main():
    print("="*80)
    print("  MULTI-DATABASE TRAFFIC VISUALIZATION")
    print("="*80)
    print()
    print("Starting 3 Flask servers:")
    print("  🔵 Neo4j      → http://localhost:5000")
    print("  🟢 ArangoDB   → http://localhost:5001")
    print("  🐅 TigerGraph → http://localhost:5002")
    print()
    print("="*80)
    print()
    
    script_dir = Path(__file__).parent
    
    servers = [
        {
            "name": "Neo4j",
            "script": script_dir / "neo4j_viewer.py",
            "port": 5000,
            "color": "🔵"
        },
        {
            "name": "ArangoDB",
            "script": script_dir / "arangodb_viewer.py",
            "port": 5001,
            "color": "🟢"
        },
        {
            "name": "TigerGraph",
            "script": script_dir / "tigergraph_viewer.py",
            "port": 5002,
            "color": "🐅"
        }
    ]
    
    processes = []
    
    try:
        # Start each server
        for server in servers:
            print(f"{server['color']} Starting {server['name']} server on port {server['port']}...")
            
            # Start process in background
            proc = subprocess.Popen(
                [sys.executable, str(server['script'])],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            processes.append({
                "name": server['name'],
                "process": proc,
                "port": server['port'],
                "color": server['color']
            })
            
            time.sleep(2)  # Give it time to start
        
        print()
        print("="*80)
        print("  ALL SERVERS RUNNING!")
        print("="*80)
        print()
        print("Open these URLs in your browser:")
        for p in processes:
            print(f"  {p['color']} {p['name']:12} → http://localhost:{p['port']}")
        print()
        print("="*80)
        print()
        print("Press Ctrl+C to stop all servers")
        print()
        
        # Keep running and monitor processes
        while True:
            time.sleep(1)
            for p in processes:
                if p['process'].poll() is not None:
                    print(f"\n⚠ {p['name']} server stopped unexpectedly!")
                    # Try to get error output
                    stderr = p['process'].stderr.read()
                    if stderr:
                        print(f"Error: {stderr}")
    
    except KeyboardInterrupt:
        print("\n")
        print("="*80)
        print("  STOPPING ALL SERVERS...")
        print("="*80)
        print()
        
        for p in processes:
            try:
                print(f"{p['color']} Stopping {p['name']}...")
                p['process'].terminate()
                p['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"  ⚠ Force killing {p['name']}...")
                p['process'].kill()
            except Exception as e:
                print(f"  ⚠ Error stopping {p['name']}: {e}")
        
        print()
        print("All servers stopped.")
        print()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up processes
        for p in processes:
            try:
                p['process'].terminate()
            except:
                pass
        
        sys.exit(1)

if __name__ == '__main__':
    main()