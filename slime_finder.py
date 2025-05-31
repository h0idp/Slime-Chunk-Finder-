import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from math import sqrt, ceil
from concurrent.futures import ThreadPoolExecutor
import time
import json
from numba import cuda, int64, float64
import threading
import sys
import os
import atexit
import random


# ==== Optimized Configuration ====
CHUNK_SIZE = 16
SPHERE_RADIUS_BLOCKS = 128
SPHERE_RADIUS_CHUNKS = SPHERE_RADIUS_BLOCKS // CHUNK_SIZE
GPU_THREADS_PER_BLOCK = 512  # Optimized for RTX 4070
MAX_BATCH_SIZE = 1024 * 1024  # 1M chunks per batch

# === Optimized GPU Kernels ===
@cuda.jit
def gpu_find_best_area_kernel(seed, scan_radius, sphere_radius_chunks, chunk_size, 
                             best_counts, best_centers, sphere_radius_blocks):
    """Optimized kernel that finds the best area directly on GPU using the Java algorithm"""
    idx = cuda.grid(1)
    scan_size = 2 * scan_radius + 1
    total_positions = scan_size * scan_size
    
    if idx >= total_positions:
        return
    
    # Convert linear index to coordinates
    cx = (idx % scan_size) - scan_radius
    cz = (idx // scan_size) - scan_radius
    
    count = 0
    center_bx = cx * chunk_size + 8
    center_bz = cz * chunk_size + 8
    
    # Search for slime chunks in the spherical area   
    for dx in range(-sphere_radius_chunks, sphere_radius_chunks + 1):
        for dz in range(-sphere_radius_chunks, sphere_radius_chunks + 1):
            scx = cx + dx
            scz = cz + dz
            bx = scx * chunk_size + 8
            bz = scz * chunk_size + 8
            
            # Check if within radius
            dist_sq = (bx - center_bx)**2 + (bz - center_bz)**2
            if dist_sq <= sphere_radius_blocks * sphere_radius_blocks:
                # Calculate if it's a slime chunk using the Java algorithm
                rnd_seed = (seed + 
                         (scx * scx * 0x4c1906) +
                         (scx * 0x5ac0db) +
                         (scz * scz) * 0x4307a7 +
                         (scz * 0x5f24f))
                rnd_seed ^= 0x3ad8025f
                
                # Implement Java Random.nextInt(10)
                rnd_seed = (rnd_seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
                rnd_seed = (rnd_seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
                val = rnd_seed >> 17
                
                if (val % 10) == 0:
                    count += 1
    
    best_counts[idx] = count
    best_centers[idx * 2] = cx
    best_centers[idx * 2 + 1] = cz

@cuda.jit
def gpu_generate_slime_map_kernel(seed, center_x, center_z, radius_chunks, chunk_size, 
                                 sphere_radius_blocks, slime_map):
    """Kernel to generate the slime chunk map"""
    idx = cuda.grid(1)
    map_size = 2 * radius_chunks + 1
    total_chunks = map_size * map_size
    
    if idx >= total_chunks:
        return
    
    # Convert index to map coordinates
    local_x = idx % map_size
    local_z = idx // map_size
    
    # Global chunk coordinates
    cx = center_x + local_x - radius_chunks
    cz = center_z + local_z - radius_chunks
    
    # Block coordinates
    bx = cx * chunk_size + 8
    bz = cz * chunk_size + 8
    center_bx = center_x * chunk_size + 8
    center_bz = center_z * chunk_size + 8
    
    # Check if within radius
    dist_sq = (bx - center_bx)**2 + (bz - center_bz)**2
    if dist_sq <= sphere_radius_blocks * sphere_radius_blocks:
        # Calculate if it's a slime chunk
        rnd_seed = seed + cx * cx * 4987142 + cx * 5947611 + cz * cz * 4392871 + cz * 389711
        rnd_seed ^= 987234911
        rnd_seed = (rnd_seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
        rnd_seed = (rnd_seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        val = rnd_seed >> 17
        slime_map[idx] = 1 if (val % 10) == 0 else 0
    else:
        slime_map[idx] = -1  # Outside the radius

# === Optimized CPU Functions ===
def java_random(seed):
    """Emulates the behavior of Java's random number generator."""
    class JavaRandom:
        def __init__(self, seed):
            self.seed = seed & ((1 << 48) - 1)

        def next(self, bits):
            self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
            return self.seed >> (48 - bits)

        def next_int(self, bound):
            if bound <= 0:
                raise ValueError("Bound must be positive")
            if (bound & (bound - 1)) == 0:  # Power of 2
                return (bound * self.next(31)) >> 31
            bits, val = self.next(31), 0
            while bits - (val := bits % bound) + (bound - 1) < 0:
                bits = self.next(31)
            return val

    return JavaRandom(seed)

def is_slime_chunk_java(world_seed, x_position, z_position):
    """
    Determines if a chunk is a slime chunk based on the world seed and coordinates,
    emulating the exact behavior of Java.

    Args:
        world_seed (int): The Minecraft world seed
        x_position (int): Chunk X coordinate
        z_position (int): Chunk Z coordinate

    Returns:
        bool: True if it is a slime chunk, False otherwise
    """
    # Calculate the seed following the same logic as Java
    seed_calculation = (
        world_seed +
        (x_position * x_position * 0x4c1906) +
        (x_position * 0x5ac0db) +
        (z_position * z_position * 0x4307a7) +
        (z_position * 0x5f24f)
    ) ^ 0x3ad8025f

    # Ensure the seed is within the 48-bit range
    seed_calculation &= ((1 << 48) - 1)

    # Create the random number generator with the calculated seed
    rnd = java_random(seed_calculation)

    # Return True if the first random number between 0-9 is 0
    return rnd.next_int(10) == 0

def is_slime_chunk_optimized(seed, cx, cz):
    """Optimized version that matches Java's implementation exactly."""
    # Replicate the exact calculation from Random.java
    rnd_seed = (seed + 
              (cx * cx * 0x4c1906) +  # In Java: 0x4c1906 = 4987142
              (cx * 0x5ac0db) +       # In Java: 0x5ac0db = 5947611
              (cz * cz) * 0x4307a7 +  # In Java: 0x4307a7 = 4392871
              (cz * 0x5f24f))         # In Java: 0x5f24f = 389711
    rnd_seed ^= 0x3ad8025f           # In Java: 0x3ad8025f = 987234911

    # Implement Java Random.nextInt(10)
    rnd_seed = (rnd_seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    rnd_seed = (rnd_seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)

    # In Java, nextInt(n) returns (int)(rnd.next(31) % n)
    return ((rnd_seed >> 17) % 10) == 0

def cpu_find_best_area_optimized(seed, scan_radius):
    """Optimized CPU version with improved parallelization."""
    def process_position(args):
        cx, cz = args
        count = 0
        center_bx = cx * CHUNK_SIZE + 8
        center_bz = cz * CHUNK_SIZE + 8

        for dx in range(-SPHERE_RADIUS_CHUNKS, SPHERE_RADIUS_CHUNKS + 1):
            for dz in range(-SPHERE_RADIUS_CHUNKS, SPHERE_RADIUS_CHUNKS + 1):
                scx = cx + dx
                scz = cz + dz
                bx = scx * CHUNK_SIZE + 8
                bz = scz * CHUNK_SIZE + 8

                dist_sq = (bx - center_bx)**2 + (bz - center_bz)**2
                if dist_sq <= SPHERE_RADIUS_BLOCKS * SPHERE_RADIUS_BLOCKS:
                    if is_slime_chunk_optimized(seed, scx, scz):
                        count += 1
        return (count, cx, cz)

    positions = [(x, z) for x in range(-scan_radius, scan_radius + 1)
                        for z in range(-scan_radius, scan_radius + 1)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_position, positions))

    best = max(results, key=lambda x: x[0])
    return best[0], (best[1], best[2])

def gpu_find_best_area_optimized(seed, scan_radius):
    """Fully optimized GPU version"""
    scan_size = 2 * scan_radius + 1
    total_positions = scan_size * scan_size
    
    # Prepare arrays
    best_counts = np.zeros(total_positions, dtype=np.int64)
    best_centers = np.zeros(total_positions * 2, dtype=np.int64)
    
    # Transfer to GPU
    d_best_counts = cuda.to_device(best_counts)
    d_best_centers = cuda.to_device(best_centers)
    
    # Configure kernel
    threads_per_block = GPU_THREADS_PER_BLOCK
    blocks = (total_positions + threads_per_block - 1) // threads_per_block
    
    # Run kernel
    gpu_find_best_area_kernel[blocks, threads_per_block](
        seed, scan_radius, SPHERE_RADIUS_CHUNKS, CHUNK_SIZE,
        d_best_counts, d_best_centers, SPHERE_RADIUS_BLOCKS
    )
    
    # Retrieve results
    d_best_counts.copy_to_host(best_counts)
    d_best_centers.copy_to_host(best_centers)
    
    # Find the best
    best_idx = np.argmax(best_counts)
    best_count = best_counts[best_idx]
    best_center = (best_centers[best_idx * 2], best_centers[best_idx * 2 + 1])
    
    return best_count, best_center

def gpu_generate_slime_map(seed, center_chunk):
    """Generates slime chunk map using GPU"""
    cx, cz = center_chunk
    map_size = 2 * SPHERE_RADIUS_CHUNKS + 1
    total_chunks = map_size * map_size
    
    slime_map = np.zeros(total_chunks, dtype=np.int32)
    d_slime_map = cuda.to_device(slime_map)
    
    threads_per_block = GPU_THREADS_PER_BLOCK
    blocks = (total_chunks + threads_per_block - 1) // threads_per_block
    
    gpu_generate_slime_map_kernel[blocks, threads_per_block](
        seed, cx, cz, SPHERE_RADIUS_CHUNKS, CHUNK_SIZE,
        SPHERE_RADIUS_BLOCKS, d_slime_map
    )
    
    d_slime_map.copy_to_host(slime_map)
    return slime_map.reshape(map_size, map_size)

class SlimeFinderGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Slime Chunk Finder")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)  # Set minimum size
        
        # Variables
        self.seed_var = tk.StringVar(value="")
        self.radius_var = tk.StringVar(value="32")
        self.result_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.use_gpu_var = tk.BooleanVar(value=cuda.is_available())
        self.coord_var = tk.StringVar()  # New variable for coordinates
        self.current_center = None  # To store the current center
        self.cancel_requested = False # To handle cancellations

        # Thread control variables
        self.running_threads = []
        self.thread_executor = None
        self.is_closing = False
        self.setup_ui()
        self.update_gpu_status()
        self.setup_cleanup()
        
        # Configure mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
    def setup_cleanup(self):
        """Configures resource cleanup on close."""
        # Window close protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Register cleanup function for atexit
        atexit.register(self.cleanup_resources)
        
    def cleanup_resources(self):
        """Cleans up all resources before closing."""
        try:
            print("Cleaning up resources...")
            
            # Mark as closing
            self.is_closing = True
            
            # Close ThreadPoolExecutor if it exists
            if self.thread_executor:
                self.thread_executor.shutdown(wait=False)
            
            # Wait for active threads to finish (with timeout)
            for thread in self.running_threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            
            # Clean CUDA context if using GPU
            if cuda.is_available():
                try:
                    cuda.close()
                    cuda.cuda.cuDeviceReset()
                except:
                    pass
            
            # Close matplotlib
            plt.close('all')
            
            print("Resources cleaned up successfully")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def on_closing(self):
        """Handles the window close event."""
        try:
            # Ask for confirmation if there are running processes
            if any(thread.is_alive() for thread in self.running_threads):
                if not messagebox.askokcancel("Exit", "There are running processes. Do you still want to exit?"):
                    return
            
            # Clean up resources
            self.cleanup_resources()
            
            # Destroy window
            self.root.quit()
            self.root.destroy()
            
            # Force process exit
            os._exit(0)
            
        except Exception as e:
            print(f"Error on closing: {e}")
            # Force exit in case of error
            os._exit(1)
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left control panel (fixed width)
        control_frame = ttk.LabelFrame(main_frame, text="Configuration", padding=10, width=250)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)  # Keep fixed width
        
        # Seed
        ttk.Label(control_frame, text="Seed:").pack(anchor=tk.W)
        seed_frame = ttk.Frame(control_frame)
        seed_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(seed_frame, textvariable=self.seed_var, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(seed_frame, text="Random", command=self.generate_random_seed, width=8).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Scan radius
        ttk.Label(control_frame, text="Scan radius (chunks):").pack(anchor=tk.W)
        radius_frame = ttk.Frame(control_frame)
        radius_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(radius_frame, textvariable=self.radius_var, width=10).pack(side=tk.LEFT)
        ttk.Label(radius_frame, text="(Recommended: 16-64)").pack(side=tk.LEFT, padx=(5, 0))
        
        # Options
        options_frame = ttk.LabelFrame(control_frame, text="Options", padding=5)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(options_frame, text="Use GPU", variable=self.use_gpu_var, 
                       command=self.update_gpu_status).pack(anchor=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(button_frame, text="🔍 Find Best Area", command=self.search_async, style="Accent.TButton").pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="❌ Cancel Search", command=self.cancel_search).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="💾 Save Coordinates", command=self.save_coordinates).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="📂 Load Coordinates", command=self.load_coordinates).pack(fill=tk.X)
        
        # Performance information
        perf_frame = ttk.LabelFrame(control_frame, text="Performance", padding=5)
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(perf_frame, textvariable=self.status_var, wraplength=200).pack()
        ttk.Label(perf_frame, textvariable=self.progress_var, wraplength=200).pack()
        
        # Results
        result_frame = ttk.LabelFrame(control_frame, text="Results", padding=5)
        result_frame.pack(fill=tk.X, expand=True)
        result_label = ttk.Label(result_frame, textvariable=self.result_var, 
                               wraplength=200, justify=tk.LEFT, font=("Consolas", 9))
        result_label.pack(fill=tk.BOTH, expand=True)
        
        # Visualization panel (with minimum size)
        viz_frame = ttk.LabelFrame(main_frame, text="Slime Chunk Map", padding=5)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        viz_frame.grid_columnconfigure(0, weight=1)
        viz_frame.grid_rowconfigure(1, weight=1)
        
        # Label for coordinates
        coord_label = ttk.Label(viz_frame, textvariable=self.coord_var, font=("Consolas", 9))
        coord_label.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 5))
        
        # Container frame for the canvas
        canvas_frame = ttk.Frame(viz_frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar (always visible)
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        
        # Toolbar buttons (using grid for better control)
        ttk.Button(toolbar_frame, text="🔍+", command=self.zoom_in).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(toolbar_frame, text="🔍-", command=self.zoom_out).grid(row=0, column=1, padx=(0, 5))
        toolbar_frame.grid_columnconfigure(3, weight=1)  # Flexible space
        ttk.Button(toolbar_frame, text="💾 PNG", command=self.save_map).grid(row=0, column=4)
        
    def update_gpu_status(self):
        if cuda.is_available() and self.use_gpu_var.get():
            gpu_name = cuda.get_current_device().name.decode('utf-8')
            status = f"✅ GPU: {gpu_name}"
        else:
            status = "⚙️ CPU (Multithreaded)"
        self.status_var.set(status)
        
    def generate_random_seed(self):
        import random
        self.seed_var.set(str(random.randint(-2147483648, 2147483647)))

    def cancel_search(self):
        """Requests immediate cancellation of the ongoing search"""
        self.cancel_requested = True
        self.progress_var.set("⏹️ Canceling...")
        
    def search_async(self):
        """Executes the search in a separate thread"""
        if self.is_closing:
            return
        
        self.cancel_requested = False
        thread = threading.Thread(target=self.search, daemon=True)
        self.running_threads.append(thread)
        thread.start()
        
    def search(self):
        try:
            if self.is_closing or self.cancel_requested:
                return
                
            seed = int(self.seed_var.get())
            radius = int(self.radius_var.get())
            
            if radius > 128:
                messagebox.showwarning("Warning", "Radius too large, it may take a long time")
            
            self.progress_var.set("🔄 Processing...")
            start_time = time.time()
               # Check if closing or cancelled
            if self.is_closing or self.cancel_requested:
                return
            
            # Execute search
            if cuda.is_available() and self.use_gpu_var.get():
               # For GPU: check cancellation before and after
                if not self.cancel_requested and not self.is_closing:
                    count, center = gpu_find_best_area_optimized(seed, radius)
                    method = "GPU"
                else:
                    return
            else:                # For CPU: check cancellation during the process
                count, center = self.cpu_find_with_cancellation(seed, radius)
                method = "CPU"
            
             # Check if cancelled before showing results
            if self.cancel_requested or self.is_closing:
                self.progress_var.set("⏹️ Search canceled")
                return
            
            elapsed = time.time() - start_time
              # Update results
            cx, cz = center
            bx, bz = cx * CHUNK_SIZE + 8, cz * CHUNK_SIZE + 8
            
            result_text = f"""🎯 BEST LOCATION:
Chunk: ({cx}, {cz})
Coordinates: ({bx}, {bz})
Slime Chunks: {count}
Radius: {SPHERE_RADIUS_BLOCKS} blocks

⚡ Performance:
Method: {method}
Time: {elapsed:.2f}s
Chunks scanned: {(2*radius+1)**2:,}"""
            
            self.result_var.set(result_text)
            self.progress_var.set(f"✅ Completed in {elapsed:.2f}s")
              # Generate and show map
            if not self.is_closing:
                self.draw_optimized_map(seed, center)
            
        except ValueError:
            if not self.is_closing:
                messagebox.showerror("Error", "Please enter valid numeric values")
                self.progress_var.set("❌ Error")
        except Exception as e:
            if not self.is_closing:
                messagebox.showerror("Error", f"Unexpected error: {str(e)}")
                self.progress_var.set("❌ Error")
    
    def on_mouse_move(self, event):
        """Handles mouse movement events and updates coordinates"""
        if event.inaxes and self.current_center:
            cx, cz = self.current_center
            center_idx = SPHERE_RADIUS_CHUNKS

            # Use exact mouse coordinates
            chunk_x = cx + event.xdata - center_idx
            chunk_z = cz + event.ydata - center_idx

            # Calculate exact block coordinates (without rounding the chunk first)
            block_x = int(chunk_x * CHUNK_SIZE + 8)
            block_z = int(chunk_z * CHUNK_SIZE + 8)

            # Calculate the chunk based on block coordinates
            chunk_x_rounded = block_x >> 4  # Equivalent to dividing by 16 and rounding down
            chunk_z_rounded = block_z >> 4

            # Update the coordinate text
            self.coord_var.set(
                f"Chunk: ({chunk_x_rounded}, {chunk_z_rounded}) | " +
                f"Block: ({block_x}, {block_z})"
            )
        else:
            self.coord_var.set("")  # Clear when the mouse is outside the map
    
    def cpu_find_with_cancellation(self, seed, scan_radius):
        """CPU version with cancellation check at each step"""
        def process_position(args):
            if self.cancel_requested or self.is_closing:
                return (0, 0, 0)
                
            cx, cz = args
            count = 0
            center_bx = cx * CHUNK_SIZE + 8
            center_bz = cz * CHUNK_SIZE + 8
            
            for dx in range(-SPHERE_RADIUS_CHUNKS, SPHERE_RADIUS_CHUNKS + 1):
                if self.cancel_requested or self.is_closing:
                    return (0, 0, 0)
                    
                for dz in range(-SPHERE_RADIUS_CHUNKS, SPHERE_RADIUS_CHUNKS + 1):
                    if self.cancel_requested or self.is_closing:
                        return (0, 0, 0)
                        
                    scx = cx + dx
                    scz = cz + dz
                    bx = scx * CHUNK_SIZE + 8
                    bz = scz * CHUNK_SIZE + 8
                    
                    dist_sq = (bx - center_bx)**2 + (bz - center_bz)**2
                    if dist_sq <= SPHERE_RADIUS_BLOCKS * SPHERE_RADIUS_BLOCKS:
                        if is_slime_chunk_optimized(seed, scx, scz):
                            count += 1
            return (count, cx, cz)
        
        positions = [(x, z) for x in range(-scan_radius, scan_radius + 1)
                            for z in range(-scan_radius, scan_radius + 1)]
        
        results = []
        
        # Process each position with constant cancellation check
        for pos in positions:
            if self.cancel_requested or self.is_closing:
                return (0, (0, 0))

            results.append(process_position(pos))
        
        # Find the best non-cancelled result
        best = max(results, key=lambda x: x[0])
        return best[0], (best[1], best[2])

    def draw_optimized_map(self, seed, center_chunk):
        """Draw the optimized map using GPU if available"""
        if self.is_closing or self.cancel_requested:
            return
            
        self.ax.clear()
        self.current_center = center_chunk  # Save current center
        
        if cuda.is_available() and self.use_gpu_var.get():
            slime_map = gpu_generate_slime_map(seed, center_chunk)
        else:
            # CPU optimized version
            cx, cz = center_chunk
            r = SPHERE_RADIUS_CHUNKS
            slime_map = np.zeros((2*r+1, 2*r+1), dtype=int)
            
            for i in range(2*r+1):
                for j in range(2*r+1):
                    if self.is_closing:
                        return
                    chunk_x = cx + (j - r)
                    chunk_z = cz + (i - r)
                    bx = chunk_x * CHUNK_SIZE + 8
                    bz = chunk_z * CHUNK_SIZE + 8
                    
                    dist = sqrt((bx - (cx*CHUNK_SIZE + 8))**2 + (bz - (cz*CHUNK_SIZE + 8))**2)
                    if dist <= SPHERE_RADIUS_BLOCKS:
                        slime_map[i, j] = 1 if is_slime_chunk_optimized(seed, chunk_x, chunk_z) else 0
                    else:
                        slime_map[i, j] = -1
    
        if not cuda.is_available() or not self.use_gpu_var.get():
            # CPU optimized version with cancellation
            cx, cz = center_chunk
            r = SPHERE_RADIUS_CHUNKS
            slime_map = np.zeros((2*r+1, 2*r+1), dtype=int)
            
            for i in range(2*r+1):
                if self.cancel_requested or self.is_closing:
                    return
                for j in range(2*r+1):
                    if self.cancel_requested or self.is_closing:
                        return
    
        if self.is_closing:
            return
    
        # Create RGB image
        img = np.zeros((*slime_map.shape, 3))
        img[slime_map == 1] = [0, 0.8, 0]      # Green for slime chunks
        img[slime_map == 0] = [0.9, 0.9, 0.9]  # Light gray for normal chunks
        img[slime_map == -1] = [0.1, 0.1, 0.1] # Dark gray for outside radius
        
        # Mark center
        center_idx = SPHERE_RADIUS_CHUNKS
        img[center_idx, center_idx] = [1, 0, 0]  # Red for center
        
        img = np.flipud(img)
        self.ax.imshow(img, interpolation='nearest')
        self.ax.set_title(f"Slime Chunks Map - Seed: {seed}", fontsize=12, fontweight='bold')
        
        # Add circle showing the radius
        circle = patches.Circle((center_idx, center_idx), SPHERE_RADIUS_CHUNKS, 
                                linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        self.ax.add_patch(circle)
        # Configure axes and grid
        self.ax.set_xlim(-0.5, slime_map.shape[1]-0.5)
        self.ax.set_ylim(-0.5, slime_map.shape[0]-0.5)
        
        # Add grid lines for chunks
        for x in range(-1, slime_map.shape[1]+1):
            self.ax.axvline(x=x-0.5, color='black', linewidth=0.5, alpha=0.3)
        for y in range(-1, slime_map.shape[0]+1):
            self.ax.axhline(y=y-0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Configure aspect ratio for perfect square chunks
        self.ax.set_aspect('equal')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=[0, 0.8, 0], label='Slime Chunk'),
            Patch(facecolor=[0.9, 0.9, 0.9], label='Normal Chunk'),
            Patch(facecolor=[1, 0, 0], label='Center'),
            Patch(facecolor='none', edgecolor='red', linestyle='--', label=f'Radius ({SPHERE_RADIUS_BLOCKS}b)')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        if not self.is_closing:
            self.canvas.draw()
    
    def zoom_in(self):
        if self.is_closing:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        factor = 0.8
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        width = (xlim[1] - xlim[0]) * factor
        height = (ylim[1] - ylim[0]) * factor
        self.ax.set_xlim(center_x - width/2, center_x + width/2)
        self.ax.set_ylim(center_y - height/2, center_y + height/2)
        self.canvas.draw()
    
    def zoom_out(self):
        if self.is_closing:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        factor = 1.25
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        width = (xlim[1] - xlim[0]) * factor
        height = (ylim[1] - ylim[0]) * factor
        self.ax.set_xlim(center_x - width/2, center_x + width/2)
        self.ax.set_ylim(center_y - height/2, center_y + height/2)
        self.canvas.draw()
    
    def save_map(self):
        if self.is_closing:
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
            title="Save map"
        )
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Map saved as {filename}")
    
    def save_coordinates(self):
        if self.is_closing:
            return
        if not self.result_var.get():
            messagebox.showwarning("Warning", "No coordinates to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save coordinates"
        )
        if filename:
            data = {
                "seed": self.seed_var.get(),
                "radius": self.radius_var.get(),
                "result": self.result_var.get(),
                "timestamp": time.time()
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Success", f"Coordinates saved to {filename}")

    def load_coordinates(self):
        if self.is_closing:
            return
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load coordinates"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                self.seed_var.set(data.get("seed", ""))
                self.radius_var.set(data.get("radius", ""))
                if "result" in data:
                    self.result_var.set(data["result"])
                messagebox.showinfo("Success", "Coordinates loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
        finally:
            self.cleanup_resources()

if __name__ == "__main__":
    try:
        app = SlimeFinderGUI()
        app.run()
    except KeyboardInterrupt:
        print("Application interrupted by the user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Ensure the process completely terminates
        try:
            os._exit(0)
        except:
            sys.exit(0)