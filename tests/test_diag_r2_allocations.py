import unittest
from rtpa_research.allocation_trace import summarize_allocations


class AllocationTraceTests(unittest.TestCase):
    def test_recycled_final_output_address_was_previously_scratch(self):
        events=[{'action':'alloc','addr':1,'size':100},
                {'action':'alloc','addr':2,'size':50},
                {'action':'free_completed','addr':1},
                {'action':'alloc','addr':1,'size':20},
                {'action':'free_completed','addr':2}]
        row=summarize_allocations(events,{1})
        self.assertEqual(row['max_transient_excluding_returned_storage_from_trace'],150)
        self.assertEqual(row['max_new_allocated_from_trace'],150)
        self.assertEqual(row['allocation_lifetimes'],3)

    def test_free_requested_is_not_completed(self):
        events=[{'action':'alloc','addr':1,'size':100},
                {'action':'free_requested','addr':1},
                {'action':'alloc','addr':2,'size':20},
                {'action':'free_completed','addr':1}]
        self.assertEqual(summarize_allocations(events,{2})['max_new_allocated_from_trace'],120)


if __name__=='__main__':unittest.main()
