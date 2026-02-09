#!/usr/bin/env python3
################################################################################
# cbir_gui.py
# Author: Krushna Sanjay Sharma
# Description: Graphical User Interface for Content-Based Image Retrieval System
#              Provides easy image selection, feature database building, and
#              visual comparison of matching results.
#
# Usage: Run from project root directory
#        python gui/cbir_gui.py
#
# Directory Structure:
#   bin/Release/          - Executables (buildFeatureDB.exe, queryImage.exe)
#   bin/data/images/      - Image database
#   bin/data/features/    - Feature CSV files (created by GUI)
#
# Requirements: Python 3.7+, tkinter, Pillow
#
# Date: February 2026
################################################################################

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
import os
import sys
from pathlib import Path

class CBIRGui:
    """
    Content-Based Image Retrieval GUI Application
    
    Features:
    - Image selection and preview
    - Feature database building (single or all)
    - Query execution with automatic CSV/metric selection
    - Top-N result visualization in grid layout
    
    Author: Krushna Sanjay Sharma
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR System - Image Retrieval")
        self.root.geometry("1200x900")
        
        # Configuration - Paths relative to project root
        self.exe_dir = "../bin/Release"
        self.image_dir = "../bin/data/images"
        self.features_dir = "../bin/data/features"
        
        # Ensure features directory exists
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Feature configuration with paths in features directory
        self.features_config = {
            'baseline': {
                'csv': os.path.join(self.features_dir, 'baseline_features.csv'),
                'metric': 'ssd',
                'description': 'Baseline 7x7 Center Square'
            },
            'histogram': {
                'csv': os.path.join(self.features_dir, 'histogram_features.csv'),
                'metric': 'histogram',
                'description': 'RGB Color Histogram'
            },
            'chromaticity': {
                'csv': os.path.join(self.features_dir, 'chromaticity_features.csv'),
                'metric': 'histogram',
                'description': 'RG Chromaticity Histogram'
            },
            'multihistogram': {
                'csv': os.path.join(self.features_dir, 'multi_features.csv'),
                'metric': 'multiregion',
                'description': 'Multi-Region Histogram (2×2 Grid)'
            },
            'texturecolor': {
                'csv': os.path.join(self.features_dir, 'texture_features.csv'),
                'metric': 'weighted',
                'description': 'Sobel Texture + Color'
            },
            'gabor': {
                'csv': os.path.join(self.features_dir, 'gabor_features.csv'),
                'metric': 'gabor',
                'description': 'Gabor Texture + Color (Advanced)'
            },
            'dnn': {
                'csv': os.path.join(self.features_dir, 'ResNet18_olym.csv'),
                'metric': 'cosine',
                'description': 'ResNet18 DNN Embeddings (Pre-computed)'
            },
            'productmatcher': {
                'csv': os.path.join(self.features_dir, 'product_features.csv'),
                'metric': 'productmatcher',
                'description': 'Task 7: DNN(60%) + Center-Color(40%)'
            },
            'faceaware': {
                'csv': os.path.join(self.features_dir, 'faceaware_features.csv'),
                'metric': 'faceaware',
                'description': 'Extension: Adaptive Face-Aware Features'
            }
        }
        
        # State
        self.selected_image_path = None
        self.query_results = []
        self.result_images = []
        
        # Verify executables exist
        self.verify_setup()
        
        # Build UI
        self.create_ui()
    
    def verify_setup(self):
        """Verify that executables and directories exist"""
        build_exe = os.path.join(self.exe_dir, 'buildFeatureDB.exe')
        query_exe = os.path.join(self.exe_dir, 'queryImage.exe')
        
        if not os.path.exists(build_exe):
            messagebox.showerror(
                "Setup Error",
                f"buildFeatureDB.exe not found at:\n{build_exe}\n\n"
                f"Please build the project first using build.bat"
            )
            sys.exit(1)
        
        if not os.path.exists(query_exe):
            messagebox.showerror(
                "Setup Error",
                f"queryImage.exe not found at:\n{query_exe}\n\n"
                f"Please build the project first using build.bat"
            )
            sys.exit(1)
        
        if not os.path.exists(self.image_dir):
            messagebox.showwarning(
                "Setup Warning",
                f"Image directory not found:\n{self.image_dir}\n\n"
                f"Please add images to this directory."
            )
        
        print(f"Setup verified:")
        print(f"  Executables: {self.exe_dir}")
        print(f"  Images: {self.image_dir}")
        print(f"  Features: {self.features_dir}")
        
    def create_ui(self):
        """Create the main user interface"""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # ============================================================
        # Section 1: Image Selection
        # ============================================================
        selection_frame = ttk.LabelFrame(main_frame, text="1. Image Selection", padding="10")
        selection_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(selection_frame, text="Select Query Image", 
                  command=self.select_image).grid(row=0, column=0, padx=5)
        
        self.image_label = ttk.Label(selection_frame, text="No image selected")
        self.image_label.grid(row=0, column=1, padx=10)
        
        # ============================================================
        # Section 2: Feature & Metric Configuration
        # ============================================================
        config_frame = ttk.LabelFrame(main_frame, text="2. Feature Configuration", padding="10")
        config_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Feature dropdown
        ttk.Label(config_frame, text="Feature Type:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.feature_var = tk.StringVar(value='histogram')
        feature_dropdown = ttk.Combobox(config_frame, textvariable=self.feature_var,
                                       values=list(self.features_config.keys()),
                                       state='readonly', width=20)
        feature_dropdown.grid(row=0, column=1, padx=5)
        feature_dropdown.bind('<<ComboboxSelected>>', self.on_feature_changed)
        
        # Feature description
        self.feature_desc_label = ttk.Label(config_frame, text="", foreground="blue")
        self.feature_desc_label.grid(row=0, column=2, padx=10)
        
        # Distance metric (read-only, auto-determined)
        ttk.Label(config_frame, text="Distance Metric:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.metric_var = tk.StringVar(value='histogram')
        metric_entry = ttk.Entry(config_frame, textvariable=self.metric_var,
                                state='readonly', width=20)
        metric_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Top N selection
        ttk.Label(config_frame, text="Top N Results:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.topn_var = tk.StringVar(value='9')
        ttk.Spinbox(config_frame, from_=1, to=20, textvariable=self.topn_var,
                   width=18).grid(row=2, column=1, padx=5)
        
        # Initialize feature description
        self.on_feature_changed(None)
        
        # ============================================================
        # Section 3: Database Building
        # ============================================================
        build_frame = ttk.LabelFrame(main_frame, text="3. Database Building", padding="10")
        build_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(build_frame, text="Build All Features", 
                  command=self.build_all_databases).grid(row=0, column=0, padx=5)
        
        ttk.Button(build_frame, text="Build Database (Selected Feature)", 
                  command=self.build_selected_database).grid(row=0, column=1, padx=5)
        
        self.build_status_label = ttk.Label(build_frame, text="Ready to build databases", 
                                           foreground="gray")
        self.build_status_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # ============================================================
        # Section 4: Query Execution
        # ============================================================
        query_frame = ttk.LabelFrame(main_frame, text="4. Query Execution", padding="10")
        query_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(query_frame, text="🔍 Find Matching Images", 
                  command=self.find_matches).grid(row=0, column=0, padx=5, pady=10)
        
        self.query_status_label = ttk.Label(query_frame, text="Select image and click to query",
                                           foreground="gray")
        self.query_status_label.grid(row=1, column=0, pady=5)
        
        # ============================================================
        # Section 5: Results Display
        # ============================================================
        results_frame = ttk.LabelFrame(main_frame, text="5. Matching Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Scrollable canvas for results
        canvas = tk.Canvas(results_frame, bg='white')
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        self.results_inner_frame = ttk.Frame(canvas)
        
        self.results_inner_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
    def on_feature_changed(self, event):
        """Update metric and description when feature changes"""
        feature = self.feature_var.get()
        config = self.features_config[feature]
        
        # Update metric (read-only)
        self.metric_var.set(config['metric'])
        
        # Update description
        self.feature_desc_label.config(text=config['description'])
    
    def select_image(self):
        """Open file dialog to select query image"""
        filepath = filedialog.askopenfilename(
            initialdir=self.image_dir,
            title="Select Query Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.selected_image_path = filepath
            filename = os.path.basename(filepath)
            self.image_label.config(text=f"Selected: {filename}")
            
            # Clear previous results
            self.clear_results()
            
            # Show preview of selected image
            self.show_query_preview()
    
    def show_query_preview(self):
        """Display preview of query image in results area"""
        self.clear_results()
        
        try:
            # Header
            header = ttk.Label(self.results_inner_frame, 
                             text="Query Image Preview",
                             font=('Arial', 12, 'bold'))
            header.grid(row=0, column=0, pady=10)
            
            # Load and display query image
            img = Image.open(self.selected_image_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            img_label = ttk.Label(self.results_inner_frame, image=photo)
            img_label.image = photo
            img_label.grid(row=1, column=0, pady=10)
            
            # Filename
            filename_label = ttk.Label(self.results_inner_frame,
                                      text=os.path.basename(self.selected_image_path),
                                      font=('Arial', 10))
            filename_label.grid(row=2, column=0)
            
        except Exception as e:
            print(f"Error showing preview: {e}")
    
    def build_all_databases(self):
        """Build feature databases for all feature types"""
        self.build_status_label.config(text="Building all databases... Please wait...", 
                                      foreground="orange")
        self.root.update()
        
        success_count = 0
        failed = []
        
        for feature_name, config in self.features_config.items():
            print(f"\nBuilding {feature_name} database...")
            success = self.build_database(feature_name, config['csv'])
            if success:
                success_count += 1
            else:
                failed.append(feature_name)
        
        if failed:
            msg = f"Built {success_count}/{len(self.features_config)} databases.\nFailed: {', '.join(failed)}"
            self.build_status_label.config(text=msg, foreground="orange")
            messagebox.showwarning("Partial Success", msg)
        else:
            self.build_status_label.config(
                text=f"All {success_count} databases built successfully!", 
                foreground="green"
            )
            messagebox.showinfo("Success", 
                               f"All {success_count} feature databases built successfully!")
    
    def build_selected_database(self):
        """Build database for selected feature only"""
        feature = self.feature_var.get()
        config = self.features_config[feature]
        
        self.build_status_label.config(
            text=f"Building {feature} database... Please wait...", 
            foreground="orange"
        )
        self.root.update()
        
        success = self.build_database(feature, config['csv'])
        
        if success:
            self.build_status_label.config(
                text=f"Database '{os.path.basename(config['csv'])}' built successfully!", 
                foreground="green"
            )
            messagebox.showinfo("Success", f"{feature} database built successfully!")
        else:
            self.build_status_label.config(
                text=f"Failed to build {feature} database", 
                foreground="red"
            )
            messagebox.showerror("Error", f"Failed to build {feature} database.\nCheck console for details.")
    
    def build_database(self, feature_type, csv_path):
        """
        Execute buildFeatureDB.exe to create feature database
        
        Args:
            feature_type: Feature type identifier
            csv_path: Full path to output CSV file
            
        Returns:
            bool: True if successful
        """
        try:
            # Get just the CSV filename (exe expects path relative to its location)
            csv_filename = os.path.relpath(csv_path, self.exe_dir)
            
            # Command to execute
            cmd = [
                os.path.join(self.exe_dir, 'buildFeatureDB.exe'),
                os.path.relpath(self.image_dir, self.exe_dir),  # Relative to exe
                feature_type,
                csv_filename
            ]
            
            print(f"Executing: {' '.join(cmd)}")
            
            # Run from exe directory
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.exe_dir
            )
            
            # Print output for debugging
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode == 0:
                print(f"✓ Built {feature_type} database successfully")
                return True
            else:
                print(f"✗ Error building {feature_type}: returncode={result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout building {feature_type} database")
            return False
        except Exception as e:
            print(f"✗ Exception building {feature_type}: {e}")
            return False
    
    def find_matches(self):
        """Execute query and display results"""
        # Validate image selected
        if not self.selected_image_path:
            messagebox.showerror("Error", "Please select a query image first!")
            return
        
        # Get configuration
        feature = self.feature_var.get()
        config = self.features_config[feature]
        csv_file = config['csv']
        metric = config['metric']
        top_n = int(self.topn_var.get())
        
        # Check if CSV exists
        if not os.path.exists(csv_file):
            msg = f"Database '{os.path.basename(csv_file)}' not found!\n\nPlease build the database first."
            messagebox.showerror("Database Not Found", msg)
            return
        
        # Update status
        self.query_status_label.config(
            text=f"Querying with {feature}... Please wait...",
            foreground="orange"
        )
        self.root.update()
        
        # Execute query
        try:
            # Get paths relative to exe directory
            query_image_rel = os.path.relpath(self.selected_image_path, self.exe_dir)
            csv_file_rel = os.path.relpath(csv_file, self.exe_dir)
            
            cmd = [
                os.path.join(self.exe_dir, 'queryImage.exe'),
                query_image_rel,
                csv_file_rel,
                feature,
                metric,
                str(top_n)
            ]
            
            print(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.exe_dir
            )
            
            if result.stdout:
                print(result.stdout)
            
            if result.returncode == 0:
                # Parse results
                self.parse_and_display_results(result.stdout, feature, metric)
            else:
                self.query_status_label.config(
                    text="Query failed!", 
                    foreground="red"
                )
                print("STDERR:", result.stderr)
                messagebox.showerror("Query Error", 
                                    f"Query failed!\n\nCheck console for details.")
                
        except subprocess.TimeoutExpired:
            self.query_status_label.config(text="Query timeout!", foreground="red")
            messagebox.showerror("Timeout", "Query took too long!")
        except Exception as e:
            self.query_status_label.config(text="Query error!", foreground="red")
            messagebox.showerror("Error", f"Query error: {e}")
    
    def parse_and_display_results(self, output, feature, metric):
        """
        Parse query output and display results
        
        Expected output format:
        ========================================
        Query Results
        ========================================
        
        Rank  Filename                      Distance
        ----------------------------------------
        1     pic.0535.jpg                  0.0000
        2     pic.0123.jpg                  0.1234
        ...
        """
        lines = output.split('\n')
        
        # Find results section
        results = []
        in_results = False
        
        for line in lines:
            if 'Rank' in line and 'Filename' in line and 'Distance' in line:
                in_results = True
                continue
            
            if in_results and line.strip():
                if line.startswith('='):
                    break
                    
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        rank = int(parts[0])
                        filename = parts[1]
                        distance = float(parts[2])
                        results.append({
                            'rank': rank,
                            'filename': filename,
                            'distance': distance
                        })
                    except:
                        pass
        
        if results:
            self.query_status_label.config(
                text=f"Found {len(results)} matches using {feature}",
                foreground="green"
            )
            self.display_results(results, feature, metric)
        else:
            self.query_status_label.config(text="No results found", foreground="red")
            messagebox.showwarning("No Results", "No matching images found!")
    
    def display_results(self, results, feature, metric):
        """Display results in grid layout"""
        # Clear previous results
        self.clear_results()
        
        # Header info
        header_frame = ttk.Frame(self.results_inner_frame)
        header_frame.grid(row=0, column=0, columnspan=3, pady=10)
        
        ttk.Label(header_frame, 
                 text=f"Query: {os.path.basename(self.selected_image_path)}", 
                 font=('Arial', 11, 'bold')).grid(row=0, column=0, padx=15)
        
        ttk.Label(header_frame, 
                 text=f"Feature: {feature}", 
                 font=('Arial', 10)).grid(row=0, column=1, padx=15)
        
        ttk.Label(header_frame, 
                 text=f"Metric: {metric}", 
                 font=('Arial', 10)).grid(row=0, column=2, padx=15)
        
        ttk.Label(header_frame, 
                 text=f"Results: {len(results)}", 
                 font=('Arial', 10)).grid(row=0, column=3, padx=15)
        
        # Display results in grid (3 columns)
        cols = 3
        img_size = (220, 220)
        
        for idx, result in enumerate(results):
            row = (idx // cols) + 1
            col = idx % cols
            
            # Create frame for each result
            result_frame = ttk.Frame(self.results_inner_frame, relief=tk.RIDGE, borderwidth=2)
            result_frame.grid(row=row, column=col, padx=10, pady=10)
            
            # Construct image path
            img_path = os.path.join(self.image_dir, result['filename'])
            
            try:
                # Load image
                img = Image.open(img_path)
                img.thumbnail(img_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Image label
                img_label = ttk.Label(result_frame, image=photo)
                img_label.image = photo  # Keep reference
                img_label.grid(row=0, column=0, padx=5, pady=5)
                
                # Rank badge
                rank_color = "green" if result['rank'] == 1 else "blue"
                rank_label = ttk.Label(result_frame, 
                                      text=f"Rank {result['rank']}", 
                                      font=('Arial', 10, 'bold'),
                                      foreground=rank_color)
                rank_label.grid(row=1, column=0, pady=2)
                
                # Filename
                ttk.Label(result_frame, 
                         text=result['filename'], 
                         font=('Arial', 9)).grid(row=2, column=0)
                
                # Distance
                distance_text = f"Distance: {result['distance']:.4f}"
                if result['rank'] == 1:
                    distance_text += " (Query)"
                ttk.Label(result_frame, 
                         text=distance_text,
                         font=('Arial', 8),
                         foreground="gray").grid(row=3, column=0, pady=2)
                
                self.result_images.append(photo)  # Keep reference
                
            except Exception as e:
                ttk.Label(result_frame, 
                         text=f"Error loading\n{result['filename']}",
                         foreground="red").grid(row=0, column=0, padx=5, pady=5)
                print(f"Error loading image {img_path}: {e}")
    
    def clear_results(self):
        """Clear previous results"""
        for widget in self.results_inner_frame.winfo_children():
            widget.destroy()
        self.result_images.clear()

def main():
    """Main entry point"""
    print("="*60)
    print("CBIR System - Graphical User Interface")
    print("Author: Krushna Sanjay Sharma")
    print("="*60)
    print()
    
    # Check current directory
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")
    print()
    
    # Verify we can find required directories
    required_paths = {
        'Executables': 'bin/Release',
        'Images': 'bin/data/images',
        'Features': 'bin/data/features'
    }
    
    all_ok = True
    for name, path in required_paths.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (NOT FOUND)")
            all_ok = False
    
    print()
    
    if not all_ok:
        print("WARNING: Some paths not found. GUI may not work correctly.")
        print("Please ensure you're running from the project root directory.")
        print()
    
    # Create and run GUI
    root = tk.Tk()
    app = CBIRGui(root)
    
    print("GUI launched. Close the window to exit.")
    print("="*60)
    
    root.mainloop()

if __name__ == "__main__":
    main()
