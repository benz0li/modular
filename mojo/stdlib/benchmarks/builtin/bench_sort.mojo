# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from random import *

from benchmark import Bench, BenchConfig, Bencher, BenchId
from stdlib.builtin.sort import (
    _heap_sort,
    _insertion_sort,
    _small_sort,
    _SortWrapper,
    sort,
)

# ===-----------------------------------------------------------------------===#
# Benchmark Utils
# ===-----------------------------------------------------------------------===#


@always_inline
fn randomize_list[
    dt: DType
](mut list: List[Scalar[dt]], size: Int, max: Scalar[dt] = Scalar[dt].MAX):
    @parameter
    if dt.is_integral():
        randint(list.unsafe_ptr(), size, 0, Int(max))
    else:
        for i in range(size):
            var res = random_float64()
            # GCC doesn't support cast from float64 to float16
            list[i] = res.cast[DType.float32]().cast[dt]()


@always_inline
fn insertion_sort[dtype: DType](mut list: List[Scalar[dtype]]):
    @parameter
    fn _less_than(
        lhs: _SortWrapper[Scalar[dtype]], rhs: _SortWrapper[Scalar[dtype]]
    ) -> Bool:
        return lhs.data < rhs.data

    _insertion_sort[_less_than](list)


@always_inline
fn small_sort[size: Int, dtype: DType](mut list: List[Scalar[dtype]]):
    @parameter
    fn _less_than(
        lhs: _SortWrapper[Scalar[dtype]], rhs: _SortWrapper[Scalar[dtype]]
    ) -> Bool:
        return lhs.data < rhs.data

    _small_sort[size, Scalar[dtype], _less_than](list.unsafe_ptr())


@always_inline
fn heap_sort[dtype: DType](mut list: List[Scalar[dtype]]):
    @parameter
    fn _less_than(
        lhs: _SortWrapper[Scalar[dtype]], rhs: _SortWrapper[Scalar[dtype]]
    ) -> Bool:
        return lhs.data < rhs.data

    _heap_sort[_less_than](list)


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a tiny list size
# ===-----------------------------------------------------------------------===#


fn bench_tiny_list_sort[dtype: DType](mut m: Bench) raises:
    alias small_list_size = 5

    @parameter
    for count in range(2, small_list_size + 1):

        @parameter
        fn bench_sort_list(mut b: Bencher) raises:
            seed(1)
            var list = List(length=count, fill=Scalar[dtype]())

            @always_inline
            @parameter
            fn preproc():
                randomize_list(list, count)

            @always_inline
            @parameter
            fn call_fn():
                sort(list)

            b.iter_preproc[call_fn, preproc]()
            _ = list^

        @parameter
        fn bench_small_sort(mut b: Bencher) raises:
            seed(1)
            var list = List(length=count, fill=Scalar[dtype]())

            @always_inline
            @parameter
            fn preproc():
                randomize_list(list, count)

            @always_inline
            @parameter
            fn call_fn():
                small_sort[count](list)

            b.iter_preproc[call_fn, preproc]()
            _ = list^

        @parameter
        fn bench_insertion_sort(mut b: Bencher) raises:
            seed(1)
            var list = List(length=count, fill=Scalar[dtype]())

            @always_inline
            @parameter
            fn preproc():
                randomize_list(list, count)

            @always_inline
            @parameter
            fn call_fn():
                insertion_sort(list)

            b.iter_preproc[call_fn, preproc]()
            _ = list^

        m.bench_function[bench_sort_list](
            BenchId(String("std_sort_random_", count, "_", dtype))
        )
        m.bench_function[bench_small_sort](
            BenchId(String("sml_sort_random_", count, "_", dtype))
        )
        m.bench_function[bench_insertion_sort](
            BenchId(String("ins_sort_random_", count, "_", dtype))
        )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a small list size
# ===-----------------------------------------------------------------------===#


fn bench_small_list_sort[dtype: DType](mut m: Bench, count: Int) raises:
    @parameter
    fn bench_sort_list(mut b: Bencher) raises:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    @parameter
    fn bench_insertion_sort(mut b: Bencher) raises:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            insertion_sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    m.bench_function[bench_sort_list](
        BenchId(String("std_sort_random_", count, "_", dtype))
    )
    m.bench_function[bench_insertion_sort](
        BenchId(String("ins_sort_random_", count, "_", dtype))
    )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a large list size
# ===-----------------------------------------------------------------------===#


fn bench_large_list_sort[dtype: DType](mut m: Bench, count: Int) raises:
    @parameter
    fn bench_sort_list(mut b: Bencher) raises:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    @parameter
    fn bench_heap_sort(mut b: Bencher) raises:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            heap_sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    m.bench_function[bench_sort_list](
        BenchId(String("std_sort_random_", count, "_", dtype))
    )

    m.bench_function[bench_heap_sort](
        BenchId(String("heap_sort_random_", count, "_", dtype))
    )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with low delta lists
# ===-----------------------------------------------------------------------===#


fn bench_low_cardinality_list_sort(mut m: Bench, count: Int, delta: Int) raises:
    @parameter
    fn bench_sort_list(mut b: Bencher) raises:
        seed(1)
        var list = List(length=count, fill=UInt8())

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count, delta)

        @always_inline
        @parameter
        fn call_fn():
            sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    @parameter
    fn bench_heap_sort(mut b: Bencher) raises:
        seed(1)
        var list = List(length=count, fill=UInt8())

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count, delta)

        @always_inline
        @parameter
        fn call_fn():
            heap_sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    m.bench_function[bench_sort_list](
        BenchId(String("std_sort_low_card_", count, "_delta_", delta))
    )
    m.bench_function[bench_heap_sort](
        BenchId(String("heap_sort_low_card_", count, "_delta_", delta))
    )


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#


def main():
    var m = Bench(BenchConfig(max_runtime_secs=0.1))

    alias dtypes = [
        DType.uint8,
        DType.uint16,
        DType.float16,
        DType.uint32,
        DType.float32,
        DType.uint64,
        DType.float64,
    ]
    var small_counts = [10, 20, 32, 64, 100]
    var large_counts = [2**12, 2**16, 2**20]
    var deltas = [0, 2, 5, 20, 100]

    @parameter
    for i in range(len(dtypes)):
        alias dtype = dtypes[i]
        bench_tiny_list_sort[dtype](m)

    @parameter
    for i in range(len(dtypes)):
        alias dtype = dtypes[i]
        for count1 in small_counts:
            bench_small_list_sort[dtype](m, count1)

    @parameter
    for i in range(len(dtypes)):
        alias dtype = dtypes[i]
        for count2 in large_counts:
            bench_large_list_sort[dtype](m, count2)

    for count3 in large_counts:
        for delta2 in deltas:
            bench_low_cardinality_list_sort(m, count3, delta2)

    m.dump_report()
