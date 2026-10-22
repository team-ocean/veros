"""Host callback threads can allocate local arrays while tracing numerical code."""

from concurrent.futures import ThreadPoolExecutor


def test_tracing_context_initializes_callback_thread_allocation_defaults():
    from veros.routines import CURRENT_CONTEXT, tracing_context
    from veros.variables import allocate

    def callback():
        assert allocate(dict(xt=8, yt=10), ("xt", "yt")).shape == (12, 14)
        assert CURRENT_CONTEXT.is_dist_safe and not CURRENT_CONTEXT.tracing
        with tracing_context():
            array = allocate(dict(xt=8, yt=10), ("xt", "yt"))
            assert CURRENT_CONTEXT.is_dist_safe and CURRENT_CONTEXT.tracing
        assert not CURRENT_CONTEXT.tracing
        return array.shape

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(callback).result() == (12, 14)


def test_tracing_context_preserves_existing_distribution_and_nested_tracing():
    from veros.routines import CURRENT_CONTEXT, tracing_context

    def callback():
        CURRENT_CONTEXT.is_dist_safe = False
        assert not getattr(CURRENT_CONTEXT, "tracing", False)
        with tracing_context():
            assert not CURRENT_CONTEXT.is_dist_safe
            with tracing_context():
                assert CURRENT_CONTEXT.tracing
            assert CURRENT_CONTEXT.tracing
        assert not CURRENT_CONTEXT.tracing and not CURRENT_CONTEXT.is_dist_safe

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(callback).result()


def test_host_callback_routine_stack_and_distribution_policy_are_thread_local():
    from veros.routines import CURRENT_CONTEXT, enter_routine
    from veros.variables import allocate

    main_level = CURRENT_CONTEXT.routine_stack.stack_level
    original_policy = CURRENT_CONTEXT.is_dist_safe
    token = object()

    def callback():
        assert CURRENT_CONTEXT.is_dist_safe
        assert CURRENT_CONTEXT.routine_stack.stack_level == 0
        assert allocate(dict(xt=8, yt=10), ("xt", "yt")).shape == (12, 14)
        with enter_routine("callback", token):
            assert CURRENT_CONTEXT.routine_stack.stack_level == 1
        assert CURRENT_CONTEXT.routine_stack.stack_level == 0

    try:
        with enter_routine("main", token):
            CURRENT_CONTEXT.is_dist_safe = False
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(callback).result()
            assert not CURRENT_CONTEXT.is_dist_safe
            assert CURRENT_CONTEXT.routine_stack.stack_level == main_level + 1
    finally:
        CURRENT_CONTEXT.is_dist_safe = original_policy
    assert CURRENT_CONTEXT.routine_stack.stack_level == main_level
