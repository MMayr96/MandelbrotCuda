#include <vector>
#include <future>

class task_group final {
public:
	explicit task_group() = default;

	task_group(task_group const&) = delete;
	task_group(task_group&&) = default;

	~task_group() {
		join_all();
	}

	task_group& operator =(task_group const&) = delete;
	task_group& operator =(task_group&&) = default;

	template <typename fun_t, typename ...args_t>
	void add(fun_t&& fun, args_t&& ...args) {
		m_group.push_back(
			std::async(std::launch::async, std::forward<fun_t>(fun), std::forward<args_t>(args)...)
		);
	}

	void join_all() {
		for (auto& f : m_group) f.wait();
	}

private:
	std::vector<std::future<void>> m_group;
};

class thread_group final {
public:
	explicit thread_group() = default;

	thread_group(thread_group const&) = delete;
	thread_group(thread_group&&) = default;

	~thread_group() {
		join_all();
	}

	thread_group& operator =(thread_group const&) = delete;
	thread_group& operator =(thread_group&&) = default;

	template <typename fun_t, typename ...args_t>
	void add(fun_t&& fun, args_t&& ...args) {
		m_group.emplace_back(std::forward<fun_t>(fun), std::forward<args_t>(args)...);
	}

	void join_all() {
		for (auto& t : m_group) if (t.joinable()) t.join();
	}

private:
	std::vector<std::thread> m_group;
};