# EC2 Ubuntu Server Setup Guide for Valkey Performance Testing

Complete guide for setting up an EC2 Ubuntu instance for running Valkey server with comprehensive performance profiling capabilities.

## Prerequisites
- Fresh Ubuntu EC2 instance (ARM recommended)
- Root/sudo access

## 1. Initial System Setup

```bash
# Set hostname
sudo hostnamectl set-hostname oss-cme

# Update system
sudo apt-get update
sudo apt update
```

## 2. Install Core Development Tools

```bash
# Essential build tools
sudo apt install -y \
  build-essential \
  git \
  make \
  cmake \
  pkg-config \
  g++ \
  clang-format \
  clang-tidy \
  clangd \
  ninja-build

# System libraries
sudo apt install -y \
  tcl \
  tcl-dev \
  libjemalloc-dev \
  libssl-dev \
  libsystemd-dev \
  libgtest-dev \
  libelf-dev \
  libdw-dev \
  libaudit-dev \
  libslang2-dev \
  libperl-dev \
  python3-dev \
  python3-pip \
  python3.12-venv \
  libiberty-dev \
  libnuma-dev \
  libzstd-dev \
  libcap-dev \
  libbfd-dev \
  libtraceevent-dev \
  libunwind-dev \
  systemtap-sdt-dev \
  libbabeltrace-dev \
  libcapstone-dev \
  libpfm4-dev \
  binutils-dev

# Locale support
sudo apt install -y \
  locales-all \
  locales

# Utilities
sudo apt install -y \
  lsb-release \
  curl \
  coreutils \
  gpg \
  htop \
  flex \
  bison
```

## 3. Install and Fix Perf Tools

The system perf tools often have issues. Follow these steps to get a working perf installation:

```bash
# Install base perf packages (may not work initially)
sudo apt install -y \
  linux-tools-common \
  linux-tools-generic \
  linux-tools-$(uname -r) \
  linux-tools-aws \
  linux-cloud-tools-aws
```

### Build Perf from Source (Recommended)

Create `fix-perf.sh`:

```bash
#!/bin/bash
set -e

# Clone kernel source
git clone --depth=1 --branch=v6.11 \
  https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git

cd linux/tools/perf

# Build perf
make clean
make -j$(nproc)

# Install system-wide
sudo make install

echo "Perf installed successfully"
perf --version
```

Run the script:

```bash
sudo chmod +x fix-perf.sh
sudo bash ./fix-perf.sh
```

## 4. Configure Perf Permissions

```bash
# Allow unprivileged perf access
echo kernel.perf_event_paranoid=-1 | sudo tee /etc/sysctl.d/99-perf.conf
echo kernel.kptr_restrict=0 | sudo tee -a /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

### Optional: Passwordless Perf for Convenience

```bash
sudo visudo -f /etc/sudoers.d/perf
# Add this line:
# ubuntu ALL=(ALL) NOPASSWD: /usr/bin/perf
```

## 5. Configure Memory Settings

```bash
# Enable transparent hugepages
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# Allocate 4GB of 2MB hugepages
echo 2000 | sudo tee /proc/sys/vm/nr_hugepages
```

**Note:** These settings reset on reboot. To persist, add to `/etc/rc.local` or create a systemd service.

## 6. Setup Valkey

```bash
# Create working directories
mkdir -p ~/oss ~/ec
cd ~/oss

# Clone Valkey repository (for building from source)
git clone https://github.com/valkey-io/valkey.git
```

### Build Valkey from Source (Optional)

```bash
cd ~/oss/valkey
mkdir build-release
cd build-release
cmake ..
make -j$(nproc)

# Test the build
../build-release/bin/valkey-server &
../build-release/bin/valkey-cli
```

### Setup Pre-built Binaries

```bash
cd ~/ec

# Copy your pre-built binaries and libraries:
# - valkey-server-ec-8.2
# - Configuration file (e.g., valkey-ec-cme.conf)
# - Required shared libraries (lib*.so)

# Set library path permanently
export LD_LIBRARY_PATH=/home/ubuntu/ec/:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/home/ubuntu/ec/:$LD_LIBRARY_PATH' >> ~/.bashrc

# Update system library cache
sudo ldconfig
```

## 7. Run Valkey Server

```bash
cd ~/ec
export LD_LIBRARY_PATH=/home/ubuntu/ec/:$LD_LIBRARY_PATH
./valkey-server-ec-8.2 valkey-ec-cme.conf
```

### Process Management

```bash
# Find Valkey process
ps -ef | grep valkey-server

# View threads
ps -eLo pid,ppid,tid,class,rtprio,ni,pri,psr,pcpu,stat,wchan:14,comm | grep valkey-server

# Kill Valkey
kill -9 $(pgrep valkey-server)
```

## 8. Performance Profiling

### Create Performance Statistics Script

Create `stats.sh`:

```bash
#!/bin/bash
PERF=perf
PID=$(pgrep valkey-server)

echo "=== Performance Statistics for Valkey (PID: $PID) ==="

# Basic CPU stats
echo -e "\n--- CPU Cycles & Instructions ---"
sudo $PERF stat -e cycles,instructions -p $PID -- sleep 10

# Cache stats
echo -e "\n--- Cache Performance ---"
sudo $PERF stat -e L1-dcache-load-misses,ll_cache_miss_rd,mem_access \
  -p $PID -- sleep 10

# TLB stats
echo -e "\n--- TLB Performance ---"
sudo $PERF stat -e l1d_tlb_refill,l2d_tlb_refill \
  -p $PID -- sleep 10

# Memory usage
echo -e "\n--- Memory Usage ---"
cat /proc/$PID/status | grep VmRSS
```

```bash
sudo chmod +x stats.sh
./stats.sh
```

### Live Performance Monitoring

```bash
# Interactive thread view
top -H -p $(pgrep valkey-server)

# Or use htop
htop
```

### Detailed CPU Profiling

```bash
# Record CPU cycles with call graphs (5 seconds)
sudo perf record -e '{cpu-cycles,instructions,L1-dcache-load-misses}' \
  -g --call-graph dwarf -p $(pgrep valkey-server) -- sleep 5

# View hierarchical report
sudo perf report --hierarchy
```

### TLB Miss Analysis

```bash
# Record TLB misses
sudo perf record -e dTLB-load-misses,l1d_tlb_refill \
  -g --call-graph dwarf -p $(pgrep valkey-server) sleep 10

# View report with threshold
sudo perf report --hierarchy --percent-limit 0.5
```

### Advanced Event Recording

```bash
# Record scheduler events and CPU cycles
sudo perf record -e cycles,sched:sched_switch,sched:sched_stat_sleep \
  -g -p $(pgrep valkey-server) -- sleep 30

# View specific symbol
sudo perf report --stdio --symbol=SendReply --call-graph=graph,0.5
```

### Report Formats

```bash
# Hierarchical view
sudo perf report --hierarchy

# Call graph with threshold
sudo perf report --call-graph=graph,0.5,callee

# Text output
sudo perf report --stdio --call-graph=graph,0.5,callee -F overhead,symbol
```

## 9. Available Perf Events

Check available events on your system:

```bash
perf list
```

Common ARM-specific events:
- `cpu-cycles`, `instructions`
- `L1-dcache-load-misses`
- `l1d_tlb_refill`, `l2d_tlb_refill`, `l3d_cache_refill`
- `ll_cache_miss_rd`, `mem_access`
- `ase_spec` (speculative execution)
- `branch-misses`
- `stalled-cycles-frontend`, `stalled-cycles-backend`

## 10. Troubleshooting

### Perf Not Working

If system perf is broken, use the absolute path to the version that works:

```bash
# Find available perf versions
ls /usr/lib/linux-tools-*/perf

# Use specific version
/usr/lib/linux-tools-6.8.0-85/perf record -g -p $(pgrep valkey-server)
```

Or build from source using the `fix-perf.sh` script above.

### Library Not Found Errors

```bash
# Always set LD_LIBRARY_PATH before running
export LD_LIBRARY_PATH=/home/ubuntu/ec/:$LD_LIBRARY_PATH

# Or copy libraries to system path
sudo cp ~/ec/lib*.so /usr/local/lib/
sudo ldconfig
```

### Permission Denied for Perf

```bash
# Check current setting
cat /proc/sys/kernel/perf_event_paranoid

# Should be -1 or 1 for non-root access
sudo sysctl -w kernel.perf_event_paranoid=-1
```

## 11. Persistent Configuration

Add to `~/.bashrc` for automatic setup on login:

```bash
# Library path
export LD_LIBRARY_PATH=/home/ubuntu/ec/:$LD_LIBRARY_PATH

# Perf alias (if using specific version)
alias perf='/usr/lib/linux-tools-6.8.0-85/perf'
```

## Notes

- **Memory settings** (hugepages, THP) reset on reboot - create startup scripts for persistence
- **Perf data files** accumulate in working directory - clean periodically with `rm perf.data*`
- **ARM vs x86**: Event names differ between architectures (e.g., `l2d_cache_refill` on ARM vs `LLC-loads` on x86)
- **Performance impact**: Profiling with perf adds 5-10% overhead; use appropriate sample rates
- **DWARF call graphs**: Provide complete stack traces but generate larger data files than frame pointers

## Quick Reference Commands

```bash
# Start Valkey
cd ~/ec && export LD_LIBRARY_PATH=/home/ubuntu/ec/:$LD_LIBRARY_PATH
./valkey-server-ec-8.2 valkey-ec-cme.conf

# Quick profile
sudo perf record -g --call-graph dwarf -p $(pgrep valkey-server) -- sleep 5
sudo perf report --hierarchy

# Monitor threads
top -H -p $(pgrep valkey-server)

# Statistics
./stats.sh

# Check memory
cat /proc/$(pgrep valkey-server)/status | grep VmRSS
```