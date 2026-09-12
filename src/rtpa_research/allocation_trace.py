"""CPU accounting of actual allocator events, including address reuse."""


def summarize_allocations(events,output_storage_addresses):
    live={};lifetimes={};timeline=[]
    for index,event in enumerate(events):
        action=event['action'];address=event.get('addr');size=event.get('size',0)
        if action=='alloc':
            if address in live:raise ValueError('Allocation address reused before completed free')
            generation=(address,index);live[address]=generation;lifetimes[generation]=size
            timeline.append(('alloc',generation))
        elif action=='free_completed' and address in live:
            timeline.append(('free',live.pop(address)))
    # Only the final live allocation at an output address is retained output.
    # An earlier temporary at that same recycled address remains scratch.
    outputs={live[address] for address in output_storage_addresses if address in live}
    current=set();peak_new=0;peak_scratch=0
    for action,generation in timeline:
        if action=='alloc':current.add(generation)
        else:current.remove(generation)
        peak_new=max(peak_new,sum(lifetimes[g] for g in current))
        peak_scratch=max(peak_scratch,sum(lifetimes[g] for g in current-outputs))
    return {'max_new_allocated_from_trace':peak_new,
            'max_transient_excluding_returned_storage_from_trace':peak_scratch,
            'allocation_lifetimes':len(lifetimes),'retained_output_allocation_lifetimes':len(outputs),
            'address_reuse_corrected':True}
