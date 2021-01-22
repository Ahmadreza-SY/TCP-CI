

module TestResult
	SUCCESS = 0
	EXCEPTION = 1	
	ASSERTION = 2
	UNKNOWN_FAILURE = 3
end


class TestExtractor
	ANSI_COLOR_CODES_REGEX = /\x1B\[([0-9]{1,3}((;[0-9]{1,2})?){1,2})?[mGK]/
	PROGRESS_LOGS_REGEX = /Progress \(\d*\):.*[\n\r]*/
	TestRun = Struct.new(:test_name, :test_duration, :test_result)

	def fetch_and_clean_logs(job)
		log = job.log.body
		if log.nil?
			return nil, 0
		end
		cleaned_log = log.gsub(ANSI_COLOR_CODES_REGEX, '').gsub(PROGRESS_LOGS_REGEX, '')
		received_bytes = log.bytesize
		return cleaned_log, received_bytes
	end

	def convert_match_to_test_run(match)
		return nil
	end

	def get_test_regex
		return nil
	end
	
	def extract_all_tests(job)
		log, received_bytes = fetch_and_clean_logs(job)
		unless log.nil?
			matches = log.scan(get_test_regex)
			return matches.map { |match|
				convert_match_to_test_run(match)
			}.compact, received_bytes
		end
		return [], received_bytes
	end
end

class MavenTestExtractor < TestExtractor
	def get_test_regex
		return /Running (?<test_name>([A-Za-z]{1}[A-Za-z\d_]*\.)+[A-Za-z][A-Za-z\d_]*)(.*?)Tests run: (?<total_tests>\d*), Failures: (?<failed_tests>\d*), Errors: (?<error_tests>\d*), Skipped: (?<skipped_tests>\d*), Time elapsed: (?<test_duration>[+-]?([0-9]*[.])?[0-9]+)/m
	end
	
	def convert_match_to_test_run(match)
		test_result = TestResult::SUCCESS
		failed_tests = match[2].to_i
		error_tests = match[3].to_i
		if failed_tests > 0 || error_tests > 0
			test_result = (error_tests > failed_tests) ? TestResult::EXCEPTION : TestResult::ASSERTION
		end
		return TestRun.new(
			match[0],
			(match[5].to_f * 1000).to_i,
			test_result
		)
	end
end

class GTestExtractor < TestExtractor
	def get_test_regex
		return /\[ RUN      \](.*?)((\[       (?<test_pass>OK) \])|(\[  (?<test_fail>FAILED)  \])) (?<test_name>(\w|\.)*) \((?<test_duration>\d+) ms\)/m
	end

	def convert_match_to_test_run(match)
		test_fail = match[1]
		test_result = TestResult::SUCCESS
		if test_fail == "FAILED"
			test_result = TestResult::UNKNOWN_FAILURE
		end
		return TestRun.new(
			match[2],
			match[3].to_i,
			test_result
		)
	end
end