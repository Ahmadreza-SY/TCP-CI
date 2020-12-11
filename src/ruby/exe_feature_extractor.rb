require 'travis'
require 'fileutils'
require 'parallel'
require 'rover-df'
require 'tqdm'
require 'json'

TEST_LOG_REGEX = /Running (?<test_class>([A-Za-z]{1}[A-Za-z\d_]*\.)+[A-Za-z][A-Za-z\d_]*)(.*?)Tests run: (?<total_tests>\d*), Failures: (?<failed_tests>\d*), Errors: (?<error_tests>\d*), Skipped: (?<skipped_tests>\d*), Time elapsed: (?<tests_duration>[+-]?([0-9]*[.])?[0-9]+)/m
ANSI_COLOR_CODES_REGEX = /\x1B\[([0-9]{1,3}((;[0-9]{1,2})?){1,2})?[mGK]/
PROGRESS_LOGS_REGEX = /Progress \(\d*\):.*[\n\r]*/
SCRIPT_REGEX = /The command "(?<script>.*)" exited with \d*./
TestRun = Struct.new(:total_tests, :failed_tests, :error_tests, :skipped_tests, :tests_duration, :test_class)

def extract_all_tests(job)
    log = job.log.body
    if log.nil?
        return [], 0
    end
    cleaned_log = log.gsub(ANSI_COLOR_CODES_REGEX, '').gsub(PROGRESS_LOGS_REGEX, '')
    received_bytes = log.bytesize
    script = if job.config.key?("script") 
        job.config["script"].first  
    else 
        matched = cleaned_log.match(SCRIPT_REGEX)
        if matched.nil? 
            return [], received_bytes
        end
        matched[:script]
    end
    script_log = cleaned_log.match(/travis_time:start:(\w+)[\n\r]\$ #{script}[\n\r](?<log>.*?)travis_time:end/m)
    if script_log.nil? 
        return [], received_bytes
		end
		matches = script_log["log"].scan(TEST_LOG_REGEX)
		return matches.map { |match|
			TestRun.new(
					match[1].to_i,
					match[2].to_i,
					match[3].to_i,
					match[4].to_i,
					match[5].to_f,
					match[0]
			)
    }.compact, received_bytes
end

def save_build(build, builds_file, sep)
	build_colnames = ['id', 'travis_id', 'state', 'start_time', 'duration', 'commit_hash', 'config']
	if builds_file.pos == 0
		builds_file.write("#{build_colnames.join(sep)}\n")
	end
	start_time = build.started_at.nil? ? "" : build.started_at.strftime("%Y-%m-%d %H:%M")
	build_features = [build.number, build.id, build.state, start_time, build.duration, build.commit.sha, build.config.to_json]
	builds_file.write("#{build_features.join(sep)}\n")
end

def save_job(job, jobs_file, sep)
	job_colnames = ['id', 'travis_id', 'build_id', 'state', 'start_time', 'duration', 'commit_hash', 'config']
	if jobs_file.pos == 0
		jobs_file.write("#{job_colnames.join(sep)}\n")
	end
	start_time = job.started_at.nil? ? "" : job.started_at.strftime("%Y-%m-%d %H:%M")
	job_features = [job.number, job.id, job.build.number, job.state, start_time, job.duration, job.commit.sha, job.config.to_json]
	jobs_file.write("#{job_features.join(sep)}\n")
end

def save_test_execution(test_runs, job, build, exe_file, sep)
	exe_columns = ["test_name", "build", "job", "test_result", "duration"]
	if exe_file.pos == 0
		exe_file.write("#{exe_columns.join(sep)}\n")
	end
	unless test_runs.empty?
		exe_file.write(test_runs.map {|run|
				test_result = (run.failed_tests == 0 && run.error_tests == 0) ? 0 : 1
				values = [run.test_class, build.number, job.number, test_result, (run.tests_duration * 1000).to_i]
				values.join(sep)
			}.join("\n") + "\n"
		)
	end
end

def fetch_logs_and_create_dataset(repository_slug, output_dir, concurrency=1)
    puts "Starting to get #{repository_slug} repository CI build logs"
    exe_path = "#{output_dir}/test_execution_history.csv"
    builds_path = "#{output_dir}/builds.csv"
    jobs_path = "#{output_dir}/jobs.csv"
    if File.file?(exe_path) && File.file?(builds_path) && File.file?(jobs_path)
			puts "Skipping test execution history data extraction, dataset already exists."
			return
		end
    repository = Travis::Repository.find(repository_slug)
    exe_file = File.open(exe_path, "w")
    builds_file = File.open(builds_path, "w")
    jobs_file = File.open(jobs_path, "w")
    
    received_megabytes = 0.0
    progress_message = "Initailizing ..."
    Parallel.each(repository.each_build, in_threads: concurrency, progress: progress_message) { |build|
			begin
				unless build.nil?
					save_build(build, builds_file, ';')
					build.jobs.each { |job|
						unless job.nil?
							save_job(job, jobs_file, ';')
							test_runs, received_bytes = extract_all_tests(job)
							received_megabytes += received_bytes.to_f / 2**20
							progress_message.sub!(/\A.*\Z/, "Received #{received_megabytes.round(2)} MBs of logs")
							save_test_execution(test_runs, job, build, exe_file, ',')
						end
					}
				end
			rescue StandardError => e
					sleep_seconds = 15
					puts "Exception occurred: #{e.message}, retrying in #{sleep_seconds} seconds ..."
					sleep(sleep_seconds)
					retry
			end
    }
    exe_file.close
    builds_file.close
    jobs_file.close
    puts "Finshed extracting #{repository_slug} test execution history data. Total logs received: #{received_megabytes.round(2)} MBs\n"
end

def download_logs_and_save_test_cases(repository_slug, output_dir)
    FileUtils.makedirs(output_dir)
    dataset_path = "#{output_dir}/test_execution_history.csv"
    if File.file?(dataset_path)
        puts "Skipping #{repository_slug} repository, execution history #{dataset_path} already exists."
        return
    end
    fetch_logs_and_create_dataset(repository_slug, output_dir, 8)
end

if ARGV.length != 2
	puts "Exactly two arguments (project slug and output dir) is required!"
	exit
end

download_logs_and_save_test_cases(ARGV[0], ARGV[1])