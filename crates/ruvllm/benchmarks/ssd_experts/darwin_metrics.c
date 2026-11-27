#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/types.h>
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    int pid = atoi(argv[1]);
    struct rusage_info_v4 usage = {0};
    struct xsw_usage swap = {0};
    size_t n = sizeof(swap);
    if (pid <= 0 || proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&usage) != 0 ||
        sysctlbyname("vm.swapusage", &swap, &n, NULL, 0) == -1) { perror("Darwin metrics"); return 1; }
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) return 1;
    printf("{\"os\":\"macos\",\"process_read_bytes\":%" PRIu64
           ",\"peak_footprint_bytes\":%" PRIu64 ",\"host_swap_used_bytes\":%" PRIu64 ",\"host_swapouts_pages\":%" PRIu64 "}\n",
           (uint64_t)usage.ri_diskio_bytesread, (uint64_t)usage.ri_lifetime_max_phys_footprint,
           (uint64_t)swap.xsu_used, (uint64_t)vm.swapouts);
    return 0;
}
