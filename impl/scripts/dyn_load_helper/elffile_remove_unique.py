import argparse
import shutil
import sys
from elftools.elf.elffile import ELFFile


def patch(input: str, output: str | None, *, verbose: bool = False) -> None:
    # https://llvm.org/doxygen/BinaryFormat_2ELF_8h_source.html#l01291
    # https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.symtab.html
    # https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html

    ELF64_SYM_SIZE = 24  # size of 64 bit struct.
    ST_INFO_OFFSET = 4  # size of Elf64_Word (uint32_t)
    STB_GNU_UNIQUE = 10
    STB_WEAK = 2
    STV_HIDDEN = 2

    def check(file):
        # Also we may debug with: hexdump -C -n 20 FILE_PATH
        ELFCLASS64 = 2  # 64-bit objects
        ET_DYN = 3  # shared object file

        read_int = lambda nbytes: int.from_bytes(file.read(nbytes), "little")
        file.seek(4)  # offset EI_CLASS
        assert read_int(1) == ELFCLASS64, "only x64 library are supported"
        file.seek(16)  # offset EI_NIDENT
        assert read_int(2) == ET_DYN, "only dynamic library are supported"
        file.seek(0)  # reset

    address = []  # address of st_info
    with open(input, "rb") as file:
        check(file)
        elf = ELFFile(file)
        address = []
        section = elf.get_section_by_name(
            ".dynsym"
        )  # symtab ignored as we are using dlopen
        for index, symbol in enumerate(section.iter_symbols()):
            # st_info consists of bind (higher 4 bit) and type (lower 4 bit)
            if symbol.entry.st_info.bind in ["STB_GNU_UNIQUE", "STB_LOOS"]:
                # typedef struct {
                #     Elf64_Word    st_name;
                #     unsigned char st_info; <--- here
                #     unsigned char st_other;
                #     Elf64_Half    st_shndx;
                #     Elf64_Addr    st_value;
                #     Elf64_Xword   st_size;
                # } Elf64_Sym;
                offset = (
                    section.header.sh_addr + index * ELF64_SYM_SIZE + ST_INFO_OFFSET
                )
                address.append(offset)

                if verbose:
                    print(f"Found UNIQUE 0x{address[-1]:08x}: {symbol.name}")

    print(f"Found {len(address)} symbol(s)")
    if output is None:
        print(f"Patch inplace: {input}")
    else:
        print(f"Output to: {output}")

    output = (
        input
        if output is None
        else shutil.copyfile(input, output, follow_symlinks=False)
    )
    with open(output, "r+b") as file:
        for offset in address:
            if verbose:
                print(f"Patch UNIQUE 0x{offset:08x}")

            file.seek(offset)
            info = int.from_bytes(file.read(1), "little")
            assert (info >> 4) == STB_GNU_UNIQUE

            file.seek(offset)
            file.write(bytes([(STB_WEAK << 4) | (info & 0xF), STV_HIDDEN]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace STB_GNU_UNIQUE or STB_LOOS with STB_WEAK in shared libraries."
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.set_defaults(call=patch)
    args = parser.parse_args(sys.argv[1:])
    args.call(args.input, args.output, verbose=args.verbose)
