require 'travis'
require 'fileutils'
require 'parallel'
require 'tqdm'
require 'json'
require './src/ruby/test_extractor.rb'
require './src/ruby/data_storing.rb'

module LogType
	MAVEN = 0
	GTEST = 1
end

UNIQUE_SEPARATOR = "\t"

def fetch_logs_and_create_dataset(repository_slug, test_extractor, output_dir, concurrency=1)
	puts "Starting to get #{repository_slug} repository CI build logs"
	exe_path = "#{output_dir}/test_execution_history.csv"
	builds_path = "#{output_dir}/builds.csv"
	jobs_path = "#{output_dir}/jobs.csv"
	if File.size?(exe_path) && File.size?(builds_path) && File.size?(jobs_path)
		puts "Skipping test execution history data extraction, dataset already exists."
		return
	end
	exe_file = File.open(exe_path, "w")
	builds_file = File.open(builds_path, "w")
	jobs_file = File.open(jobs_path, "w")
	
	received_megabytes = 0.0
	progress_message = "Initailizing ..."
	repository = nil
	build_numbers = nil
	begin
		repository = Travis::Repository.find(repository_slug)
		last_build_number = repository.last_build.number
		build_numbers = (1..last_build_number).to_a
	rescue StandardError => e
			sleep_seconds = 15
			puts "Exception occurred: #{e.message}, retrying in #{sleep_seconds} seconds ..."
			sleep(sleep_seconds)
			retry
	end
	Parallel.each(build_numbers, in_threads: concurrency, progress: progress_message) { |build_number|
		begin
			build = repository.build(build_number)
			unless build.nil?
				save_build(build, builds_file, UNIQUE_SEPARATOR)
				build.jobs.each { |job|
					unless job.nil?
						save_job(job, jobs_file, UNIQUE_SEPARATOR)
						test_runs, received_bytes = test_extractor.extract_all_tests(job)
						received_megabytes += received_bytes.to_f / 2**20
						progress_message.sub!(/\A.*\Z/, "Received #{received_megabytes.round(2)} MBs of logs")
						save_test_execution(test_runs, job, build, exe_file, ',')
					end
				}
			end
		rescue StandardError => e
				sleep_seconds = 15
				progress_message.sub!(/\A.*\Z/, "Exception occurred: #{e.message}, retrying in #{sleep_seconds} seconds ...")
				sleep(sleep_seconds)
				retry
		end
	}
	exe_file.close
	builds_file.close
	jobs_file.close
	puts "Finshed extracting #{repository_slug} test execution history data. Total logs received: #{received_megabytes.round(2)} MBs\n"
end

def download_logs_and_save_test_cases(repository_slug, log_type, output_dir)
	FileUtils.makedirs(output_dir)
	dataset_path = "#{output_dir}/exe.csv"
	if File.size?(dataset_path)
		puts "Skipping #{repository_slug} repository, execution history #{dataset_path} already exists."
		return
	end
	test_extractor = if log_type == LogType::MAVEN
		MavenTestExtractor.new
	elsif log_type == LogType::GTEST
		GTestExtractor.new
	end
	fetch_logs_and_create_dataset(repository_slug, test_extractor, output_dir, 8)
end

if ARGV.length != 3
	puts "Exactly three arguments (project slug, log type, and output dir) is required!"
	exit
end

download_logs_and_save_test_cases(ARGV[0], ARGV[1].to_i, ARGV[2])